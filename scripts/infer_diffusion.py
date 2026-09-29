import os
import sys
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.downscaling import (
    ConditionalWeatherDiffusion,
    GaussianDiffusionScheduler,
    ConditionalWeatherUNet,
    BilinearDownscaler,
)
from weather_core.preprocessing import WeatherPreprocessor

RAW_FILE = Path("D:/SIH26078_AERIS/data/raw/era5_amphan_2020.nc")
MODEL_DIR = Path("D:/SIH26078_AERIS/models")
OUTPUT_FILE = Path("D:/SIH26078_AERIS/data/processed/diffusion_downscaled_output.nc")


def run_diffusion_inference():
    parser = argparse.ArgumentParser(description="AERIS Conditional Diffusion Downscaling Inference Engine")
    parser.add_argument("--dataset", type=str, default=str(RAW_FILE))
    parser.add_argument("--model-path", type=str, default=str(MODEL_DIR / "diffusion_best.pt"))
    parser.add_argument("--output-path", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--device", type=str, default="auto")
    args, _ = parser.parse_known_args()

    print("=" * 80)
    print("AERIS CONDITIONAL DIFFUSION DOWNSCALING INFERENCE ENGINE")
    print("=" * 80)

    dataset_path = Path(args.dataset) if Path(args.dataset).exists() else Path("D:/SIH26078_AERIS/data/processed/india_weather_sample_standardized.nc")
    if not dataset_path.exists():
        print(f"Error: Target dataset not found at {dataset_path}")
        return

    # 1. Load dataset & preprocess
    preprocessor = WeatherPreprocessor()
    ds_std, _ = preprocessor.preprocess_dataset(str(dataset_path), dataset_name="era5_amphan_2020")

    var_name = "msl" if "msl" in ds_std.data_vars else list(ds_std.data_vars.keys())[0]
    
    # Crop to Amphan bounding box
    amphan_bbox = {"min_lat": 12.0, "max_lat": 24.0, "min_lon": 82.0, "max_lon": 92.0}
    bilinear = BilinearDownscaler(target_resolution_km=5.0)
    da_fine_target, meta = bilinear.downscale_event_region(ds_std, var_name, amphan_bbox)

    y_vals = da_fine_target.values
    if y_vals.ndim == 3:
        y_vals = y_vals[0]

    y_target_tensor = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y_target_tensor = F.interpolate(y_target_tensor, size=(64, 64), mode="bilinear", align_corners=False)

    # Coarse conditioning tensor (downsampled then interpolated back)
    coarse_cond_tensor = F.interpolate(y_target_tensor, size=(16, 16), mode="bilinear", align_corners=False)
    coarse_cond_tensor = F.interpolate(coarse_cond_tensor, size=(64, 64), mode="bilinear", align_corners=False)

    # 2. Hardware selection & Model load
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Dataset Path         : {dataset_path}")
    print(f"Variable Downscaled  : {var_name}")
    print(f"Active Compute Device: {device}")

    model = ConditionalWeatherUNet(in_channels=2, base_channels=16).to(device)
    scheduler = GaussianDiffusionScheduler(timesteps=100)
    diffusion = ConditionalWeatherDiffusion(model=model, scheduler=scheduler, device=device)

    model_path = Path(args.model_path)
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained checkpoint from {model_path} (Loss = {checkpoint.get('loss', 0.0):.6f})")
    else:
        print(f"Warning: Checkpoint not found at {model_path}. Using initialized model weights.")

    y_mean = float(np.mean(y_vals))
    y_std = float(np.std(y_vals)) + 1e-6

    # 3. Perform Reverse Diffusion Sampling
    print("--- EXECUTING CONDITIONAL REVERSE DIFFUSION SAMPLING (100 TIMESTEPS) ---")
    model.eval()
    coarse_cond = coarse_cond_tensor.to(device)
    
    with torch.no_grad():
        downscaled_sample = diffusion.sample(coarse_cond, num_steps=100)

    downscaled_array = downscaled_sample.squeeze().cpu().numpy() * y_std + y_mean


    # 4. Construct NetCDF dataset output
    lat_arr = np.linspace(amphan_bbox["min_lat"], amphan_bbox["max_lat"], 64)
    lon_arr = np.linspace(amphan_bbox["min_lon"], amphan_bbox["max_lon"], 64)

    ds_output = xr.Dataset(
        data_vars={
            f"{var_name}_downscaled": (
                ["latitude", "longitude"],
                downscaled_array,
                {
                    "long_name": f"Conditional Diffusion Downscaled {var_name}",
                    "units": ds_std[var_name].attrs.get("units", "N/A"),
                    "resolution_km": 5.0,
                    "method": "Conditional Weather UNet DDPM Diffusion",
                },
            ),
            f"{var_name}_coarse_baseline": (
                ["latitude", "longitude"],
                coarse_cond_tensor.squeeze().cpu().numpy(),
                {
                    "long_name": f"Coarse Baseline {var_name}",
                    "units": ds_std[var_name].attrs.get("units", "N/A"),
                },
            ),
        },
        coords={
            "latitude": lat_arr,
            "longitude": lon_arr,
        },
        attrs={
            "title": "AERIS Phase 10 Conditional Diffusion Downscaling Output",
            "source_dataset": "ERA5 Reanalysis (Cyclone Amphan 2020)",
            "spatial_refinement": "28 km -> 5 km (64x64 localized crop)",
            "disclaimer": "Experimental downscaling prototype; not operational weather prediction.",
        },
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_output.to_netcdf(str(out_path))

    print("\n" + "=" * 80)
    print("DIFFUSION INFERENCE COMPLETE")
    print(f"Output NetCDF Saved : {out_path}")
    print(f"Downscaled Grid Size: {downscaled_array.shape}")
    print("=" * 80)


if __name__ == "__main__":
    run_diffusion_inference()
