import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.downscaling import (
    BilinearDownscaler,
    ConditionalWeatherUNet,
    GaussianDiffusionScheduler,
    ConditionalWeatherDiffusion,
    DownscalingEvaluator,
)
from weather_core.preprocessing import WeatherPreprocessor

RAW_FILE = Path("D:/SIH26078_AERIS/data/raw/era5_amphan_2020.nc")
MODEL_DIR = Path("D:/SIH26078_AERIS/models")


def run_diffusion_visualization():
    print("=" * 80)
    print("AERIS CONDITIONAL DIFFUSION VS BASELINE DIAGNOSTIC COMPARISON")
    print("=" * 80)

    target_path = RAW_FILE if RAW_FILE.exists() else Path("D:/SIH26078_AERIS/data/processed/india_weather_sample_standardized.nc")
    if not target_path.exists():
        print(f"Error: Target dataset file not found at {target_path}")
        return

    # 1. Load dataset & preprocess
    preprocessor = WeatherPreprocessor()
    ds_std, _ = preprocessor.preprocess_dataset(str(target_path), dataset_name="era5_amphan_2020")

    var_name = "msl" if "msl" in ds_std.data_vars else list(ds_std.data_vars.keys())[0]

    # Crop to Amphan bounding box
    amphan_bbox = {"min_lat": 12.0, "max_lat": 24.0, "min_lon": 82.0, "max_lon": 92.0}

    # 2. Bilinear 5-km Baseline
    bilinear = BilinearDownscaler(target_resolution_km=5.0)
    da_fine_baseline, meta = bilinear.downscale_event_region(ds_std, var_name, amphan_bbox)

    y_baseline_vals = da_fine_baseline.values
    if y_baseline_vals.ndim == 3:
        y_baseline_vals = y_baseline_vals[0]

    baseline_tensor = torch.tensor(y_baseline_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    baseline_tensor = F.interpolate(baseline_tensor, size=(64, 64), mode="bilinear", align_corners=False)

    coarse_tensor = F.interpolate(baseline_tensor, size=(16, 16), mode="bilinear", align_corners=False)
    coarse_tensor = F.interpolate(coarse_tensor, size=(64, 64), mode="bilinear", align_corners=False)

    # 3. Diffusion Model Downscaling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalWeatherUNet(in_channels=2, base_channels=16).to(device)
    scheduler = GaussianDiffusionScheduler(timesteps=100)
    diffusion = ConditionalWeatherDiffusion(model=model, scheduler=scheduler, device=device)

    model_path = MODEL_DIR / "diffusion_best.pt"
    if model_path.exists():
        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded trained diffusion model from {model_path}")
    else:
        print("Note: Using initialized diffusion model for visual diagnostics.")

    y_mean = float(np.mean(y_baseline_vals))
    y_std = float(np.std(y_baseline_vals)) + 1e-6

    model.eval()
    with torch.no_grad():
        diffusion_sample = diffusion.sample(coarse_tensor.to(device), num_steps=100)

    diff_arr = diffusion_sample.squeeze().cpu().numpy() * y_std + y_mean
    base_arr = baseline_tensor.squeeze().cpu().numpy() * y_std + y_mean
    coarse_arr = coarse_tensor.squeeze().cpu().numpy() * y_std + y_mean
    difference_arr = diff_arr - base_arr


    # Metrics computation
    def calc_metrics(arr):
        return {
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    m_coarse = calc_metrics(coarse_arr)
    m_base = calc_metrics(base_arr)
    m_diff = calc_metrics(diff_arr)

    # 4. Comparative Metrics Table
    print("\n" + "=" * 80)
    print("EXTREME AMPLITUDE COMPARISON (COARSE VS BASELINE VS DIFFUSION)")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Coarse (28km)':<18} | {'Bilinear (5km)':<18} | {'Diffusion (5km)':<18}")
    print("-" * 80)
    print(f"{'Peak Maximum (Max)':<20} | {m_coarse['max']:<18.2f} | {m_base['max']:<18.2f} | {m_diff['max']:<18.2f}")
    print(f"{'Peak Minimum (Min)':<20} | {m_coarse['min']:<18.2f} | {m_base['min']:<18.2f} | {m_diff['min']:<18.2f}")
    print(f"{'Mean Intensity':<20} | {m_coarse['mean']:<18.2f} | {m_base['mean']:<18.2f} | {m_diff['mean']:<18.2f}")
    print(f"{'Std Deviation':<20} | {m_coarse['std']:<18.2f} | {m_base['std']:<18.2f} | {m_diff['std']:<18.2f}")
    print(f"{'95th Percentile':<20} | {m_coarse['p95']:<18.2f} | {m_base['p95']:<18.2f} | {m_diff['p95']:<18.2f}")
    print(f"{'99th Percentile':<20} | {m_coarse['p99']:<18.2f} | {m_base['p99']:<18.2f} | {m_diff['p99']:<18.2f}")

    # 5. Spatial Quality Benchmark
    eval_diff_vs_base = DownscalingEvaluator.evaluate_downscaling_quality(base_arr, diff_arr)
    print("\n--- DIFFUSION VS BASELINE SPATIAL BENCHMARK ---")
    print(f"  Spatial MAE           : {eval_diff_vs_base['spatial_mae']:.4f}")
    print(f"  Spatial RMSE          : {eval_diff_vs_base['spatial_rmse']:.4f}")
    print(f"  Pearson Correlation   : {eval_diff_vs_base['pearson_correlation']:.4f}")
    print(f"  Max Absolute Difference: {np.max(np.abs(difference_arr)):.4f}")

    print("\n" + "=" * 80)
    print("DIFFUSION DOWNSCALING DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_diffusion_visualization()
