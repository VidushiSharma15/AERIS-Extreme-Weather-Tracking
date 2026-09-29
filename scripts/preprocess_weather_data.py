import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_core.preprocessing.pipeline import WeatherPreprocessor
from weather_core.config import get_settings


def preprocess_weather_file(file_path: str, output_name: Optional[str] = None):
    path = Path(file_path)
    if not path.is_absolute():
        settings = get_settings()
        alt_path = Path(settings.sample_data_root) / file_path
        if alt_path.exists():
            path = alt_path

    print("=" * 70)
    print(f"PREPROCESSING WEATHER DATASET: {path.name}")
    print("=" * 70)

    preprocessor = WeatherPreprocessor()
    ds_std, report = preprocessor.preprocess_dataset(
        source_path_or_ds=str(path),
        dataset_name=output_name or path.stem,
    )

    settings = get_settings()
    out_dir = Path(settings.processed_data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{report.dataset_name}_standardized.nc"
    ds_std.to_netcdf(out_file, engine="netcdf4")

    print(f"\nStandardized dataset saved to: {out_file.resolve()}")
    print(f"File Size: {out_file.stat().st_size / 1024:.2f} KB")

    print("\n--- PREPROCESSING REPORT ---")
    print(json.dumps(report.to_dict(), indent=2))
    print("=" * 70)
    return str(out_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        settings = get_settings()
        sample = Path(settings.sample_data_root) / "india_weather_sample.nc"
        if sample.exists():
            preprocess_weather_file(str(sample))
        else:
            print("Usage: python scripts/preprocess_weather_data.py <file_path>")
            sys.exit(1)
    else:
        preprocess_weather_file(sys.argv[1])
