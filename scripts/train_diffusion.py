import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.downscaling import (
    SR3WeatherUNet,
    GaussianDiffusionScheduler,
    AERISCombinedLoss,
    BilinearDownscaler,
)
from weather_core.preprocessing import WeatherPreprocessor

RAW_FILE = Path("D:/SIH26078_AERIS/data/raw/era5_amphan_2020.nc")
MODEL_DIR = Path("D:/SIH26078_AERIS/models")


def run_diffusion_training():
    parser = argparse.ArgumentParser(description="AERIS SR3-Inspired Conditional Diffusion Trainer")
    parser.add_argument("--config", type=str, default="configs/model/diffusion.yaml")
    parser.add_argument("--dataset", type=str, default=str(RAW_FILE))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    args, _ = parser.parse_known_args()

    print("=" * 80)
    print("AERIS SR3-INSPIRED CONDITIONAL DIFFUSION PROTOTYPE TRAINER")
    print("=" * 80)

    target_path = Path(args.dataset) if Path(args.dataset).exists() else Path("D:/SIH26078_AERIS/data/processed/india_weather_sample_standardized.nc")
    if not target_path.exists():
        print(f"Error: Target dataset not found at {target_path}")
        return

    # 1. Load dataset & preprocess
    preprocessor = WeatherPreprocessor()
    ds_std, _ = preprocessor.preprocess_dataset(str(target_path), dataset_name="era5_amphan_2020")

    var_name = "msl" if "msl" in ds_std.data_vars else list(ds_std.data_vars.keys())[0]

    # Downscale bounding box region to 64x64 spatial grid for GPU safety
    amphan_bbox = {"min_lat": 12.0, "max_lat": 24.0, "min_lon": 82.0, "max_lon": 92.0}
    bilinear = BilinearDownscaler(target_resolution_km=5.0)
    da_fine_target, _ = bilinear.downscale_event_region(ds_std, var_name, amphan_bbox)

    y_target_vals = da_fine_target.values
    if y_target_vals.ndim == 3:
        y_target_vals = y_target_vals[0]

    y_mean = float(np.mean(y_target_vals))
    y_std = float(np.std(y_target_vals)) + 1e-6
    y_norm_vals = (y_target_vals - y_mean) / y_std

    y_target_tensor = torch.tensor(y_norm_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y_target_tensor = F.interpolate(y_target_tensor, size=(64, 64), mode="bilinear", align_corners=False)

    # Coarse conditioning tensor (downsampled 4x then interpolated)
    coarse_cond_tensor = F.interpolate(y_target_tensor, size=(16, 16), mode="bilinear", align_corners=False)
    coarse_cond_tensor = F.interpolate(coarse_cond_tensor, size=(64, 64), mode="bilinear", align_corners=False)

    # 2. Hardware & Memory Footprint Estimation
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Dataset Target File  : {target_path}")
    print(f"Variable Downscaled  : {var_name}")
    print(f"Active Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU Execution'})")

    param_size_mb = (2.5 * 1024 * 1024) / (1024 * 1024)
    tensor_size_mb = (1 * 2 * 64 * 64 * 4 * 10) / (1024 * 1024)
    est_vram_mb = param_size_mb + tensor_size_mb + 250.0
    print(f"Estimated VRAM Footprint: {est_vram_mb:.2f} MB (< 1.5 GB - Safe for RTX GPU / CPU)\n")

    # 3. Model & Loss Setup (SR3 UNet + Combined Multi-Objective Loss)
    model = SR3WeatherUNet(in_channels=2, base_channels=16).to(device)
    scheduler = GaussianDiffusionScheduler(timesteps=100)
    loss_fn = AERISCombinedLoss(
        recon_weight=1.0,
        gradient_weight=0.1,
        extreme_weight=0.1,
        physics_weight=0.05,
        spectral_weight=0.05,
        gradient_enabled=True,
        extreme_enabled=True,
        physics_enabled=True,
        spectral_enabled=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0005)

    y_target = y_target_tensor.to(device)
    coarse_cond = coarse_cond_tensor.to(device)

    # 4. Training Loop (Smoke Run: Configurable Epochs)
    print(f"--- RUNNING AERIS SR3-INSPIRED CONDITIONAL DIFFUSION TRAINING ({args.epochs} EPOCHS) ---")
    best_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        t = torch.randint(0, scheduler.timesteps, (1,), device=device).long()
        noise = torch.randn_like(y_target)
        x_noisy, target_noise = scheduler.q_sample(y_target, t, noise)

        # Forward pass: Predict noise
        pred_noise = model(x_noisy, coarse_cond, t)

        # Reconstructed field estimate for spatial loss calculation
        sqrt_alpha_bar_t = scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha_bar_t = scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(device)
        pred_field = (x_noisy - sqrt_one_minus_alpha_bar_t * pred_noise) / (sqrt_alpha_bar_t + 1e-6)

        # Multi-Objective Combined Loss
        loss_dict = loss_fn(
            pred_noise=pred_noise,
            target_noise=target_noise,
            pred_field=pred_field,
            target_field=y_target,
            coarse_cond=coarse_cond,
            variable_name=var_name,
        )
        total_loss = loss_dict["loss"]
        total_loss.backward()
        optimizer.step()

        loss_val = float(total_loss.item())
        print(
            f"  [Epoch {epoch}/{args.epochs}] Total Loss = {loss_val:.6f} | "
            f"L_recon = {loss_dict['l_recon'].item():.6f} | "
            f"L_grad = {loss_dict['l_gradient'].item():.6f} | "
            f"L_ext = {loss_dict['l_extreme'].item():.6f} | "
            f"L_phys = {loss_dict['l_physics'].item():.6f} | "
            f"L_spec = {loss_dict['l_spectral'].item():.6f}"
        )

        checkpoint = {
            "epoch": epoch,
            "architecture": "AERIS SR3-Inspired Conditional Diffusion U-Net",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_val,
            "device": str(device),
            "loss_weights": {
                "l_recon": 1.0,
                "l_gradient": 0.1,
                "l_extreme": 0.1,
                "l_physics": 0.05,
                "l_spectral": 0.05,
            },
        }
        torch.save(checkpoint, str(MODEL_DIR / "diffusion_sr3_latest.pt"))

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(checkpoint, str(MODEL_DIR / "diffusion_sr3_best.pt"))

    print("\n" + "=" * 80)
    print("AERIS SR3 DIFFUSION PROTOTYPE TRAINING COMPLETE")
    print(f"Best Loss Saved : {best_loss:.6f}")
    print(f"Checkpoint Path : {MODEL_DIR / 'diffusion_sr3_best.pt'}")
    print("=" * 80)


if __name__ == "__main__":
    run_diffusion_training()
