from pathlib import Path
from typing import List, Dict, Any, Optional
from weather_core.preprocessing import WeatherPreprocessor
from weather_core.climatology import ClimatologyEngine
from weather_core.anomaly import AnomalyEngine, SpatialRegionDetector, ExtremeWeatherEvent
from weather_core.config import get_settings


class EventService:
    """
    Service layer executing scientific anomaly detection and serving events.
    """

    def __init__(self):
        self.settings = get_settings()
        self._cached_events: List[ExtremeWeatherEvent] = []
        self._loaded_dataset_name: str = ""

    def process_and_cache_events(self, file_path: Optional[str] = None) -> List[ExtremeWeatherEvent]:
        target_path = Path(file_path) if file_path else Path(self.settings.sample_data_root) / "india_weather_sample.nc"
        if not target_path.exists():
            target_path = Path(self.settings.processed_data_root) / "india_weather_sample_standardized.nc"

        if not target_path.exists():
            self._cached_events = []
            return []

        preprocessor = WeatherPreprocessor()
        ds_std, _ = preprocessor.preprocess_dataset(str(target_path), dataset_name=target_path.stem)

        clim_engine = ClimatologyEngine(historical_dataset=ds_std)
        clim_baseline = clim_engine.compute_baseline_statistics()

        anomaly_engine = AnomalyEngine(climatology_baseline=clim_baseline)
        _, events = anomaly_engine.detect_extreme_events(ds_std, source_dataset_name=target_path.stem)

        self._cached_events = events
        self._loaded_dataset_name = target_path.name
        return self._cached_events

    def get_events(self) -> List[Dict[str, Any]]:
        import json
        amphan_json_path = Path("D:/SIH26078_AERIS/data/processed/nepsg_amphan_events.json")
        if amphan_json_path.exists():
            with open(amphan_json_path, "r") as f:
                data = json.load(f)
                events_list = data.get("events", [])
                formatted_events = []
                for evt in events_list:
                    lat = evt.get("latitude", 18.5)
                    lon = evt.get("longitude", 86.5)
                    formatted_events.append({
                        "event_id": evt.get("event_id", "NEPSG_AMPHAN_EVENT"),
                        "event_type": "cyclone_amphan",
                        "severity": evt.get("severity", "Very Severe Cyclonic Storm"),
                        "timestamp": evt.get("timestamp", "2020-05-17T00:00:00Z"),
                        "variable_name": "mean_sea_level_pressure",
                        "units": "hPa",
                        "centroid": {"latitude": lat, "longitude": lon},
                        "bounding_box": {
                            "min_lat": max(5.0, lat - 1.5),
                            "max_lat": min(35.0, lat + 1.5),
                            "min_lon": max(70.0, lon - 1.5),
                            "max_lon": min(95.0, lon + 1.5),
                        },
                        "area_km2": evt.get("area_km2", 12300.0),
                        "max_intensity": evt.get("min_msl_hpa", 915.5),
                        "mean_intensity": evt.get("min_msl_hpa", 915.5),
                        "max_z_score": evt.get("max_z_score", 4.8),
                        "mean_z_score": evt.get("mean_z_score", 3.9),
                        "affected_grid_cells": evt.get("affected_grid_cells", 12),
                        "source_dataset": "tigge-forecasts (dems NCMRWF)",
                        "provenance_metadata": {
                            "lead_time_hour": evt.get("lead_time_hour", 0),
                            "max_wind_speed_ms": evt.get("max_wind_speed_ms", 33.6),
                            "min_msl_hpa": evt.get("min_msl_hpa", 960.0),
                            "bearing_deg": evt.get("bearing_deg", 0.0),
                            "speed_kmh": evt.get("speed_kmh", 0.0),
                            "step_displacement_km": evt.get("step_displacement_km", 0.0),
                            "ensemble_support_ratio": evt.get("ensemble_support_ratio", 1.0),
                        }
                    })
                return formatted_events

        if not self._cached_events:
            self.process_and_cache_events()
        return [evt.to_dict() for evt in self._cached_events]

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        all_evts = self.get_events()
        for evt in all_evts:
            if evt.get("event_id") == event_id:
                return evt
        return None
