from typing import Dict, Any, List, Tuple, Optional
from weather_core.anomaly.spatial_regions import ExtremeWeatherEvent, BoundingBox
from .geodesic import GeodesicMath


class EventMatcher:
    """
    Computes documented composite matching scores between extreme weather events across sequential timestamps.
    """

    DEFAULT_WEIGHTS = {
        "distance": 0.40,
        "iou": 0.30,
        "intensity": 0.15,
        "area": 0.15,
    }
    MAX_SEARCH_DISTANCE_KM = 300.0  # Max distance an anomaly region can move in 1-3 hours

    @classmethod
    def calculate_bbox_iou(cls, b1: Optional[BoundingBox], b2: Optional[BoundingBox]) -> float:
        """Calculates Intersection-over-Union (IoU) of two latitude/longitude bounding boxes."""
        if b1 is None or b2 is None:
            return 0.0

        inter_min_lat = max(b1.min_lat, b2.min_lat)
        inter_max_lat = min(b1.max_lat, b2.max_lat)
        inter_min_lon = max(b1.min_lon, b2.min_lon)
        inter_max_lon = min(b1.max_lon, b2.max_lon)

        if inter_min_lat >= inter_max_lat or inter_min_lon >= inter_max_lon:
            return 0.0

        inter_area = (inter_max_lat - inter_min_lat) * (inter_max_lon - inter_min_lon)
        b1_area = (b1.max_lat - b1.min_lat) * (b1.max_lon - b1.min_lon)
        b2_area = (b2.max_lat - b2.min_lat) * (b2.max_lon - b2.min_lon)

        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    @classmethod
    def calculate_match_score(
        cls,
        event1: ExtremeWeatherEvent,
        event2: ExtremeWeatherEvent,
        max_dist_km: float = 300.0,
    ) -> float:
        """
        Calculates composite match score S in [0, 1] between event1 (time T) and event2 (time T+1).
        Returns 0.0 if event types do not match or distance > max_dist_km.
        """
        if event1.event_type != event2.event_type:
            return 0.0

        d_km = GeodesicMath.haversine_distance_km(
            event1.centroid.latitude,
            event1.centroid.longitude,
            event2.centroid.latitude,
            event2.centroid.longitude,
        )

        if d_km > max_dist_km:
            return 0.0

        s_dist = max(0.0, 1.0 - (d_km / max_dist_km))
        s_iou = cls.calculate_bbox_iou(event1.bounding_box, event2.bounding_box)

        max_i = max(abs(event1.max_intensity), abs(event2.max_intensity), 1e-6)
        s_int = max(0.0, 1.0 - (abs(event1.max_intensity - event2.max_intensity) / max_i))

        max_a = max(event1.area_km2, event2.area_km2, 1e-6)
        s_area = min(event1.area_km2, event2.area_km2) / max_a

        score = (
            cls.DEFAULT_WEIGHTS["distance"] * s_dist
            + cls.DEFAULT_WEIGHTS["iou"] * s_iou
            + cls.DEFAULT_WEIGHTS["intensity"] * s_int
            + cls.DEFAULT_WEIGHTS["area"] * s_area
        )
        return float(round(score, 4))
