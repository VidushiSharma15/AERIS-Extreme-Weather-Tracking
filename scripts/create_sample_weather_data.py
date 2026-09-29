import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weather_core.config import get_settings


def create_sample_weather_dataset(output_path: Optional[str] = None) -> str:
    settings = get_settings()
    out_dir = Path(settings.sample_data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = Path(output_path) if output_path else out_dir / "india_weather_sample.nc"

    # Define India regional spatial coordinates (0.5 degree grid resolution)
    lats = np.linspace(8.0, 36.0, num=57, dtype=np.float32)   # 8°N to 36°N
    lons = np.linspace(68.0, 96.0, num=57, dtype=np.float32)  # 68°E to 96°E

    # 5 Hourly forecast timestamps
    times = pd.date_range("2026-09-25T00:00:00", periods=5, freq="1h")

    n_time = len(times)
    n_lat = len(lats)
    n_lon = len(lons)

    # Deterministic physical weather arrays
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    temp_base = 310.0 - (lat_grid - 8.0) * 0.5  # Warmer in South, cooler in North
    temp_data = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
    for t in range(n_time):
        temp_data[t] = temp_base + np.sin(t * 0.5) * 1.5

    # Total Precipitation: 0 mm to 140 mm
    precip_data = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
    for t in range(n_time):
        center_lat = 20.0 + t * 0.2
        center_lon = 85.0 + t * 0.3
        dist_sq = (lat_grid - center_lat) ** 2 + (lon_grid - center_lon) ** 2
        precip_data[t] = np.maximum(0.0, 140.0 * np.exp(-dist_sq / 8.0))

    # Wind Speed: 2 m/s to 32 m/s
    wind_data = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
    for t in range(n_time):
        center_lat = 20.0 + t * 0.2
        center_lon = 85.0 + t * 0.3
        dist_sq = (lat_grid - center_lat) ** 2 + (lon_grid - center_lon) ** 2
        wind_data[t] = 5.0 + 27.0 * np.exp(-dist_sq / 12.0)

    # Build CF-compliant DataArrays
    ds = xr.Dataset(
        data_vars={
            "total_precipitation": xr.DataArray(
                precip_data,
                dims=["time", "latitude", "longitude"],
                attrs={
                    "long_name": "Total Accumulated Precipitation",
                    "units": "mm",
                    "standard_name": "precipitation_amount",
                },
            ),
            "2m_temperature": xr.DataArray(
                temp_data,
                dims=["time", "latitude", "longitude"],
                attrs={
                    "long_name": "2 Metre Air Temperature",
                    "units": "K",
                    "standard_name": "air_temperature",
                },
            ),
            "10m_wind_speed": xr.DataArray(
                wind_data,
                dims=["time", "latitude", "longitude"],
                attrs={
                    "long_name": "10 Metre Wind Speed",
                    "units": "m/s",
                    "standard_name": "wind_speed",
                },
            ),
        },
        coords={
            "time": ("time", times),
            "latitude": ("latitude", lats, {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", lons, {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title": "SIH26078 AERIS India Regional Weather Sample",
            "dataset_type": "DEVELOPMENT SAMPLE DATA",
            "institution": "NCMRWF / SIH 2026 AERIS Pipeline",
            "source": "CF-1.8 NetCDF Regional Meteorological Sample",
            "history": "Created for Phase 2 Data Ingestion Validation",
            "Conventions": "CF-1.8",
        },
    )

    # Save NetCDF file
    ds.to_netcdf(target_path, engine="netcdf4")
    print(f"Sample weather dataset generated successfully at: {target_path}")
    print(f"Dataset Size: {target_path.stat().st_size / 1024:.2f} KB")
    return str(target_path)


if __name__ == "__main__":
    create_sample_weather_dataset()
