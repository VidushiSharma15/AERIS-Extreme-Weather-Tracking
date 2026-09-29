import sys
from pathlib import Path
import yaml
import shutil

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.data_manager.manifest import DatasetManifest


def inspect_dataset_configs():
    configs_dir = root_dir / "configs" / "datasets"
    print("=" * 80)
    print("AERIS METEOROLOGICAL DATASET MANIFEST INSPECTION")
    print("=" * 80)

    if not configs_dir.exists():
        print(f"No dataset configurations found at: {configs_dir}")
        return

    yaml_files = list(configs_dir.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML dataset manifests found in {configs_dir}")
        return

    # Check target storage disk space on D: drive
    d_drive = Path("D:/SIH26078_AERIS/data/raw")
    d_drive.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage("D:/")
    free_gb = free / (1024 ** 3)

    print(f"Target Raw Data Directory : {d_drive}")
    print(f"Target Storage Free Space : {free_gb:.2f} GB\n")

    for yf in yaml_files:
        manifest = DatasetManifest.from_yaml(str(yf))
        print(f"--- DATASET: {manifest.dataset_name} ---")
        print(f"  Official Source        : {manifest.source}")
        print(f"  Documentation/URL      : {manifest.source_url}")
        print(f"  Dataset Type           : {manifest.dataset_type}")
        print(f"  Variables Included     : {', '.join(manifest.variables)}")
        print(f"  Physical Units         : {manifest.units}")
        print(f"  Spatial Extent (Region): Lat [{manifest.region.get('min_lat')}°N, {manifest.region.get('max_lat')}°N], "
              f"Lon [{manifest.region.get('min_lon')}°E, {manifest.region.get('max_lon')}°E]")
        print(f"  Spatial Resolution     : {manifest.resolution}")
        print(f"  Time Window            : {manifest.time_range.get('start_time')} --> {manifest.time_range.get('end_time')}")
        print(f"  Estimated Download Size: {manifest.download_size_mb:.2f} MB")
        print(f"  Target Storage Path    : {manifest.local_path}")
        print(f"  License / Access Notes : {manifest.license_notes}")
        print("-" * 80)


if __name__ == "__main__":
    inspect_dataset_configs()
