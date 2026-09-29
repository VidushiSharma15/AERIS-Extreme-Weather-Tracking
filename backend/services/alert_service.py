from typing import List, Dict, Any
from .event_service import EventService


class AlertService:
    """
    AlertService generates structured analytical alerts from scientific anomaly events.
    
    Disclaimer:
    Each alert contains explicit metadata indicating that it is an internal AERIS analytical category,
    not an official government meteorological warning.
    """

    DISCLAIMER = "AERIS internal analytical alert — not an official government warning."

    def __init__(self, event_service: EventService):
        self.event_service = event_service

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        events = self.event_service.get_events()
        alerts = []

        for idx, evt in enumerate(events, start=1):
            if evt.get("severity", "NORMAL") in ("WATCH", "MODERATE", "SEVERE"):
                alert_id = f"ALT_{evt['event_id']}_{idx:02d}"
                centroid = evt.get("centroid", {})

                alert = {
                    "alert_id": alert_id,
                    "event_id": evt["event_id"],
                    "event_type": evt["event_type"],
                    "severity": evt["severity"],
                    "latitude": centroid.get("latitude", 0.0),
                    "longitude": centroid.get("longitude", 0.0),
                    "timestamp": evt["timestamp"],
                    "anomaly_score": evt.get("max_z_score", 0.0),
                    "affected_area_km2": evt.get("area_km2", 0.0),
                    "confidence": round(min(1.0, abs(evt.get("max_z_score", 0.0)) / 5.0), 3),
                    "source_dataset": evt.get("source_dataset", "LocalNetCDF"),
                    "disclaimer": self.DISCLAIMER,
                }
                alerts.append(alert)

        return alerts
