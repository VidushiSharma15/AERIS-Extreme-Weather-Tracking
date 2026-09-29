import pandas as pd
from typing import List, Tuple
from .models import WeatherEventTrack, TrackPoint
from .geodesic import GeodesicMath


class TrajectoryCalculator:
    """
    Computes trajectory statistics, displacements, bearings, speeds, and trends for WeatherEventTrack.
    """

    @classmethod
    def update_track_statistics(cls, track: WeatherEventTrack) -> WeatherEventTrack:
        if not track.points:
            return track

        points = track.points
        track.start_time = points[0].timestamp
        track.end_time = points[-1].timestamp
        track.initial_centroid = track.initial_centroid or track.points[0].to_dict()
        p_last = points[-1]
        track.current_centroid = track.current_centroid or p_last.to_dict()

        if len(points) == 1:
            track.total_displacement_km = 0.0
            track.speed_kmh = 0.0
            track.direction_degrees = 0.0
            track.intensity_change = 0.0
            track.area_change = 0.0
            return track

        p_first = points[0]

        # Total displacement from initial point to latest point
        track.total_displacement_km = round(
            GeodesicMath.haversine_distance_km(
                p_first.latitude, p_first.longitude, p_last.latitude, p_last.longitude
            ),
            2,
        )

        # Bearing / direction from pen-ultimate to latest point
        p_prev = points[-2]
        step_dist = GeodesicMath.haversine_distance_km(
            p_prev.latitude, p_prev.longitude, p_last.latitude, p_last.longitude
        )
        track.direction_degrees = round(
            GeodesicMath.calculate_bearing_degrees(
                p_prev.latitude, p_prev.longitude, p_last.latitude, p_last.longitude
            ),
            2,
        )

        # Time difference in hours between first and last point
        try:
            t0 = pd.to_datetime(p_first.timestamp)
            t1 = pd.to_datetime(p_last.timestamp)
            dt_hours = float((t1 - t0).total_seconds() / 3600.0)
        except Exception:
            dt_hours = 1.0

        track.speed_kmh = round(
            GeodesicMath.calculate_speed_kmh(track.total_displacement_km, dt_hours), 2
        )

        # Trends
        track.intensity_change = round(p_last.intensity - p_first.intensity, 2)
        track.area_change = round(p_last.area_km2 - p_first.area_km2, 2)

        return track
