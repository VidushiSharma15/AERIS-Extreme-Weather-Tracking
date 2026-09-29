from typing import List, Dict, Any, Tuple
from .models import WeatherEventTrack, TrackPoint
from .geodesic import GeodesicMath


class BaselineTrajectoryExtrapolator:
    """
    BASELINE TRAJECTORY EXTRAPOLATION.
    Extrapolates short-term future track positions based on recent motion vector (bearing & speed).
    
    Scientific Honesty Note:
    This module performs simple, physically transparent kinematic extrapolation based on the recent velocity vector.
    It is explicitly labeled BASELINE TRAJECTORY EXTRAPOLATION and is NOT an AI/GNN prediction.
    """

    LABEL = "BASELINE TRAJECTORY EXTRAPOLATION"

    @classmethod
    def extrapolate_short_term(
        cls, track: WeatherEventTrack, lead_hours_ahead: List[float] = [1.0, 2.0, 3.0]
    ) -> List[Dict[str, Any]]:
        if not track.points or len(track.points) < 1:
            return []

        last_pt = track.points[-1]
        speed = track.speed_kmh
        bearing = track.direction_degrees

        extrapolations = []
        for lead_h in lead_hours_ahead:
            dist_km = speed * lead_h
            if dist_km == 0.0 and len(track.points) > 1:
                # Fallback to last step displacement if net speed ~ 0
                prev_pt = track.points[-2]
                dist_km = GeodesicMath.haversine_distance_km(
                    prev_pt.latitude, prev_pt.longitude, last_pt.latitude, last_pt.longitude
                )

            extrap_lat, extrap_lon = GeodesicMath.extrapolate_point(
                last_pt.latitude, last_pt.longitude, dist_km, bearing
            )

            extrapolations.append(
                {
                    "lead_hours_ahead": lead_h,
                    "extrapolated_latitude": extrap_lat,
                    "extrapolated_longitude": extrap_lon,
                    "estimated_distance_km": round(dist_km, 2),
                    "method": cls.LABEL,
                }
            )

        return extrapolations
