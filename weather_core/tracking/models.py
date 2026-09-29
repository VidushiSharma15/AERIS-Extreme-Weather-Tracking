from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from weather_core.anomaly.spatial_regions import ExtremeWeatherEvent, CentroidCoordinates, BoundingBox


@dataclass
class TrackPoint:
    timestamp: str
    latitude: float
    longitude: float
    intensity: float
    area_km2: float
    severity: str
    z_score: float
    event_id: str
    bounding_box: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.bounding_box:
            data["bounding_box"] = self.bounding_box.to_dict()
        return data


@dataclass
class WeatherEventTrack:
    track_id: str
    event_type: str
    start_time: str
    end_time: str
    points: List[TrackPoint] = field(default_factory=list)
    current_centroid: Optional[CentroidCoordinates] = None
    initial_centroid: Optional[CentroidCoordinates] = None
    total_displacement_km: float = 0.0
    direction_degrees: float = 0.0
    speed_kmh: float = 0.0
    intensity_change: float = 0.0
    area_change: float = 0.0
    confidence: float = 1.0
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.current_centroid:
            data["current_centroid"] = self.current_centroid.to_dict()
        if self.initial_centroid:
            data["initial_centroid"] = self.initial_centroid.to_dict()
        data["points"] = [p.to_dict() for p in self.points]
        return data

    def to_api_schema(self) -> Dict[str, Any]:
        """
        Formatted schema required by the API specification.
        """
        current_pos = (
            self.current_centroid.to_dict()
            if self.current_centroid
            else {"latitude": 0.0, "longitude": 0.0}
        )
        return {
            "track_id": self.track_id,
            "event_type": self.event_type,
            "severity": self.points[-1].severity if self.points else "NORMAL",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "current_position": current_pos,
            "direction_degrees": round(self.direction_degrees, 2),
            "speed_kmh": round(self.speed_kmh, 2),
            "intensity_change": round(self.intensity_change, 2),
            "area_change": round(self.area_change, 2),
            "confidence": round(self.confidence, 3),
            "trajectory": [p.to_dict() for p in self.points],
        }
