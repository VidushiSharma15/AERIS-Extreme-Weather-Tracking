import sys
import json
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from weather_core.preprocessing import WeatherPreprocessor
from weather_core.climatology import ClimatologyEngine
from weather_core.anomaly import AnomalyEngine, SpatialRegionDetector
from weather_core.tracking import (
    SpatioTemporalTracker,
    GeoJSONTrackExporter,
    BaselineTrajectoryExtrapolator,
)
from weather_core.config import get_settings


def track_weather_events_file(file_path: str, output_geojson: Optional[str] = None):
    path = Path(file_path)
    if not path.is_absolute():
        settings = get_settings()
        alt_path = Path(settings.sample_data_root) / file_path
        if not alt_path.exists():
            alt_path = Path(settings.processed_data_root) / file_path
        if alt_path.exists():
            path = alt_path

    print("=" * 75)
    print(f"RUNNING SPATIO-TEMPORAL TRACKING PIPELINE: {path.name}")
    print("=" * 75)

    # 1. Preprocess
    preprocessor = WeatherPreprocessor()
    ds_std, report = preprocessor.preprocess_dataset(str(path), dataset_name=path.stem)

    # 2. Climatology & Anomaly Detection per Timestamp
    clim_engine = ClimatologyEngine(historical_dataset=ds_std)
    clim_baseline = clim_engine.compute_baseline_statistics()

    anomaly_engine = AnomalyEngine(climatology_baseline=clim_baseline)
    ds_anomaly = anomaly_engine.compute_anomaly_fields(ds_std, climatology_baseline=clim_baseline)

    # Extract events per timestamp
    events_per_time = []
    time_steps = len(ds_std["time"]) if "time" in ds_std.dims else 1

    for t_idx in range(time_steps):
        t_ds = ds_std.isel(time=t_idx) if "time" in ds_std.dims else ds_std
        t_anom = ds_anomaly.isel(time=t_idx) if "time" in ds_anomaly.dims else ds_anomaly
        t_stamp = str(t_ds["time"].values) if "time" in t_ds.coords else "2026-09-25T00:00:00"

        t_events = []
        for var_name in list(ds_std.data_vars.keys()):
            if str(var_name).endswith("_qc_flag") or str(var_name).endswith("_anomaly") or str(var_name).endswith("_zscore"):
                continue

            z_key = f"{var_name}_zscore"
            if z_key not in t_anom:
                continue

            z_thresh = anomaly_engine.threshold_config.get(var_name).z_score_threshold
            evts = SpatialRegionDetector.detect_connected_regions(
                z_score_da=t_anom[z_key],
                val_da=t_ds[var_name],
                clim_mean_da=t_ds[var_name],
                var_name=var_name,
                timestamp=t_stamp,
                z_threshold=z_thresh,
                source_dataset=path.stem,
            )
            t_events.extend(evts)
        events_per_time.append(t_events)

    # 3. Execute SpatioTemporalTracker
    tracker = SpatioTemporalTracker(match_threshold=0.25, max_search_dist_km=400.0)
    tracks = tracker.track_events_over_time(events_per_time)

    print("\n--- SPATIO-TEMPORAL EVENT TRACKING REPORT ---")
    print(f"Total Forecast Time Steps : {time_steps}")
    print(f"Total Active Tracks       : {len(tracks)}")

    for idx, trk in enumerate(tracks, start=1):
        print(f"\n[TRACK #{idx:02d}]: {trk.track_id} ({trk.event_type})")
        print(f"  Duration              : {trk.start_time} --> {trk.end_time}")
        print(f"  Track Point Count     : {len(trk.points)}")
        print(f"  Total Displacement    : {trk.total_displacement_km} km")
        print(f"  Avg Speed             : {trk.speed_kmh} km/h")
        print(f"  Heading / Bearing     : {trk.direction_degrees} deg")
        print(f"  Intensity Trend       : {trk.intensity_change:+.2f}")
        print(f"  Area Trend            : {trk.area_change:+.2f} km^2")
        print(f"  Confidence Score      : {trk.confidence}")

        # Extrapolation
        extra = BaselineTrajectoryExtrapolator.extrapolate_short_term(trk, lead_hours_ahead=[1.0, 2.0])
        if extra:
            print(f"  [BASELINE EXTRAPOLATION (+1h)]: Lat {extra[0]['extrapolated_latitude']} N, Lon {extra[0]['extrapolated_longitude']} E")

    # 4. Save GeoJSON Trajectories
    settings = get_settings()
    out_dir = Path(settings.processed_data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_geojson = output_geojson or str(out_dir / "event_trajectories.geojson")
    GeoJSONTrackExporter.save_tracks_geojson(tracks, target_geojson)

    print(f"\nGeoJSON Event Trajectories exported to: {target_geojson}")
    print("=" * 75)
    return tracks, target_geojson


if __name__ == "__main__":
    if len(sys.argv) < 2:
        settings = get_settings()
        sample = Path(settings.sample_data_root) / "india_weather_sample.nc"
        if sample.exists():
            track_weather_events_file(str(sample))
        else:
            print("Usage: python scripts/track_weather_events.py <file_path>")
            sys.exit(1)
    else:
        track_weather_events_file(sys.argv[1])
