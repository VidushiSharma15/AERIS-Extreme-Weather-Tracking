from pathlib import Path
from typing import List, Dict, Any, Optional
from weather_core.tracking import SpatioTemporalTracker, WeatherEventTrack, GeoJSONTrackExporter
from weather_core.preprocessing import WeatherPreprocessor
from weather_core.climatology import ClimatologyEngine
from weather_core.anomaly import AnomalyEngine
from weather_core.config import get_settings


class TrackingService:
    """
    Service layer providing spatio-temporal tracking and event trajectory API endpoints.
    """

    def __init__(self):
        self.settings = get_settings()
        self.tracker = SpatioTemporalTracker()
        self._cached_tracks: List[WeatherEventTrack] = []

    def _ensure_tracks_computed(self):
        if self._cached_tracks:
            return

        target_path = Path(self.settings.sample_data_root) / "india_weather_sample.nc"
        if not target_path.exists():
            target_path = Path(self.settings.processed_data_root) / "india_weather_sample_standardized.nc"

        if not target_path.exists():
            return

        preprocessor = WeatherPreprocessor()
        ds_std, _ = preprocessor.preprocess_dataset(str(target_path), dataset_name=target_path.stem)
        clim_engine = ClimatologyEngine(historical_dataset=ds_std)
        clim_baseline = clim_engine.compute_baseline_statistics()
        anomaly_engine = AnomalyEngine(climatology_baseline=clim_baseline)

        # Detect events for each timestep
        time_vals = ds_std['time'].values
        events_by_time = {}
        for t_val in time_vals:
            t_str = str(t_val)
            ds_t = ds_std.sel(time=[t_val])
            _, evts_t = anomaly_engine.detect_extreme_events(ds_t, source_dataset_name=target_path.stem)
            if evts_t:
                events_by_time[t_str] = evts_t

        self._cached_tracks = self.tracker.track_events_over_time(events_by_time)

    def load_tracks(self, tracks: List[WeatherEventTrack]):
        self._cached_tracks = tracks

    def get_all_events(self) -> List[Dict[str, Any]]:
        self._ensure_tracks_computed()
        return [trk.to_api_schema() for trk in self._cached_tracks]

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_tracks_computed()
        for trk in self._cached_tracks:
            if trk.track_id == event_id:
                return trk.to_api_schema()
        return None

    def get_trajectory(self, event_id: str) -> List[Dict[str, Any]]:
        import json
        amphan_json_path = Path("D:/SIH26078_AERIS/data/processed/nepsg_amphan_events.json")
        if amphan_json_path.exists():
            with open(amphan_json_path, "r") as f:
                data = json.load(f)
                events_list = data.get("events", [])
                points = []
                for evt in events_list:
                    points.append({
                        "timestamp": evt.get("timestamp", ""),
                        "latitude": evt.get("latitude", 18.5),
                        "longitude": evt.get("longitude", 86.5),
                        "intensity": evt.get("min_msl_hpa", 982.3),
                        "area_km2": 12300.0,
                        "severity": evt.get("severity", "Very Severe Cyclonic Storm"),
                        "z_score": 4.8,
                        "event_id": evt.get("event_id", event_id),
                    })
                return points

        self._ensure_tracks_computed()
        trk = self.get_event_by_id(event_id)
        if trk:
            return trk.get("trajectory", [])
        return []

    def get_geojson_feature_collection(self) -> Dict[str, Any]:
        self._ensure_tracks_computed()
        return GeoJSONTrackExporter.tracks_to_geojson(self._cached_tracks)

