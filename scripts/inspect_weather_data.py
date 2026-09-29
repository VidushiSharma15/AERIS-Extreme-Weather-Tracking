import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_core.ingestion.local_netcdf import LocalNetCDFProvider
from weather_core.ingestion.validator import DatasetValidator
from weather_core.config import get_settings


def inspect_weather_file(file_path: str):
    path = Path(file_path)
    if not path.is_absolute():
        settings = get_settings()
        alt_path = Path(settings.sample_data_root) / file_path
        if alt_path.exists():
            path = alt_path

    print("=" * 70)
    print(f"WEATHER DATASET INSPECTION REPORT")
    print(f"File Path: {path.resolve()}")
    print("=" * 70)

    provider = LocalNetCDFProvider()
    ds = provider.open_dataset(str(path))
    val_report = DatasetValidator.validate_dataset(ds)
    metadata = provider.extract_metadata(str(path), dataset_name=path.stem)

    print("\n--- DATASET GLOBAL METADATA ---")
    print(f"Dataset Name       : {metadata.dataset_name}")
    print(f"Data Source        : {metadata.source}")
    print(f"Dataset Attributes : {dict(ds.attrs)}")
    print(f"File Size          : {metadata.size_bytes / 1024:.2f} KB")

    print("\n--- SPATIO-TEMPORAL DOMAIN ---")
    print(f"Latitude Coord     : {val_report['latitude_coord']} -> Range: {val_report['lat_range']} °N")
    print(f"Longitude Coord    : {val_report['longitude_coord']} -> Range: {val_report['lon_range']} °E")
    print(f"Time Coord         : {val_report['time_coord']} -> Range: {val_report['time_range']}")
    print(f"Time Steps Count   : {val_report['time_steps']}")
    print(f"Estimated Res      : {val_report['resolution_deg']}° (~{val_report['resolution_km']} km)")

    print("\n--- METEOROLOGICAL VARIABLES & STATISTICS ---")
    for var_name, info in val_report["variables"].items():
        print(f"\n  [Variable]: {var_name}")
        print(f"    Standard Units : {info['units']}")
        print(f"    Dimensions     : {info['dims']}")
        print(f"    Grid Shape     : {info['shape']}")
        print(f"    Missing Values : {info['missing_count']} / {info['total_elements']} ({info['missing_pct']}%)")
        print(f"    Min Value      : {info['min']} {info['units']}")
        print(f"    Max Value      : {info['max']} {info['units']}")
        print(f"    Mean Value     : {info['mean']} {info['units']}")

    print("\n--- SERIALIZED METADATA MANIFEST ---")
    print(json.dumps(metadata.to_dict(), indent=2))
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        settings = get_settings()
        sample = Path(settings.sample_data_root) / "india_weather_sample.nc"
        if sample.exists():
            inspect_weather_file(str(sample))
        else:
            print("Usage: python scripts/inspect_weather_data.py <file_path>")
            sys.exit(1)
    else:
        inspect_weather_file(sys.argv[1])
