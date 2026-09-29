import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_core.preprocessing import WeatherPreprocessor
from weather_core.climatology import ClimatologyEngine
from weather_core.anomaly import AnomalyEngine, GeoJSONExporter, AnomalyDiagnostics
from weather_core.config import get_settings


def detect_anomalies_file(file_path: str, output_geojson: Optional[str] = None):
    path = Path(file_path)
    if not path.is_absolute():
        settings = get_settings()
        alt_path = Path(settings.sample_data_root) / file_path
        if not alt_path.exists():
            alt_path = Path(settings.processed_data_root) / file_path
        if alt_path.exists():
            path = alt_path

    print("=" * 75)
    print(f"RUNNING ANOMALY DETECTION PIPELINE: {path.name}")
    print("=" * 75)

    # 1. Preprocess
    preprocessor = WeatherPreprocessor()
    ds_std, report = preprocessor.preprocess_dataset(str(path), dataset_name=path.stem)

    # 2. Climatology Baseline
    clim_engine = ClimatologyEngine(historical_dataset=ds_std)
    clim_baseline = clim_engine.compute_baseline_statistics()
    print(f"\n[Climatology Status]: {clim_baseline.attrs.get('climatology_status')}")

    # 3. Anomaly Engine Execution
    anomaly_engine = AnomalyEngine(climatology_baseline=clim_baseline)
    ds_anomaly, events = anomaly_engine.detect_extreme_events(
        ds=ds_std,
        climatology_baseline=clim_baseline,
        source_dataset_name=path.stem,
    )

    # 4. Print Visual Diagnostics Report
    AnomalyDiagnostics.print_event_summary(events)

    # 5. Export GeoJSON
    settings = get_settings()
    out_dir = Path(settings.processed_data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_geojson = output_geojson or str(out_dir / "detected_events.geojson")
    GeoJSONExporter.save_geojson(events, target_geojson)

    print(f"\nGeoJSON FeatureCollection exported to: {target_geojson}")
    print(f"Total GeoJSON Events Exported : {len(events)}")
    print("=" * 75)
    return events, target_geojson


if __name__ == "__main__":
    if len(sys.argv) < 2:
        settings = get_settings()
        sample = Path(settings.sample_data_root) / "india_weather_sample.nc"
        if sample.exists():
            detect_anomalies_file(str(sample))
        else:
            print("Usage: python scripts/detect_anomalies.py <file_path>")
            sys.exit(1)
    else:
        detect_anomalies_file(sys.argv[1])
