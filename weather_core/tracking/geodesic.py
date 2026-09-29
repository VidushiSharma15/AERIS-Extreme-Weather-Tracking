import math
from typing import Tuple


class GeodesicMath:
    """
    Geodesic math helper accounting for Earth's spherical geometry (WGS84 mean radius ~ 6371.0 km).
    
    Rule Enforced:
    Never use Cartesian distance sqrt((lat2-lat1)^2 + (lon2-lon1)^2) for spherical coordinates.
    """

    EARTH_RADIUS_KM = 6371.0

    @classmethod
    def haversine_distance_km(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates great-circle distance between two points in kilometers.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return cls.EARTH_RADIUS_KM * c

    @classmethod
    def calculate_bearing_degrees(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates initial bearing (heading) from point 1 to point 2 in degrees (0° to 360°).
        0° = North, 90° = East, 180° = South, 270° = West.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)

        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

        initial_bearing = math.atan2(y, x)
        initial_bearing_deg = math.degrees(initial_bearing)
        return (initial_bearing_deg + 360.0) % 360.0

    @classmethod
    def calculate_speed_kmh(cls, distance_km: float, time_diff_hours: float) -> float:
        """
        Calculates movement speed in kilometers per hour.
        """
        if time_diff_hours <= 0:
            return 0.0
        return distance_km / time_diff_hours

    @classmethod
    def extrapolate_point(
        cls, lat: float, lon: float, distance_km: float, bearing_deg: float
    ) -> Tuple[float, float]:
        """
        Extrapolates a new (latitude, longitude) given starting point, distance in km, and bearing in degrees.
        """
        r = cls.EARTH_RADIUS_KM
        d_div_r = distance_km / r
        bearing_rad = math.radians(bearing_deg)
        phi1 = math.radians(lat)
        lambda1 = math.radians(lon)

        phi2 = math.asin(
            math.sin(phi1) * math.cos(d_div_r)
            + math.cos(phi1) * math.sin(d_div_r) * math.cos(bearing_rad)
        )
        lambda2 = lambda1 + math.atan2(
            math.sin(bearing_rad) * math.sin(d_div_r) * math.cos(phi1),
            math.cos(d_div_r) - math.sin(phi1) * math.sin(phi2),
        )

        return (round(math.degrees(phi2), 4), round((math.degrees(lambda2) + 540.0) % 360.0 - 180.0, 4))
