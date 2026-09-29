from typing import List
from .models import WeatherEventTrack
from .geodesic import GeodesicMath


class TrackConfidenceCalculator:
    """
    Computes transparent analytical confidence score C in [0.1, 1.0] for a WeatherEventTrack.
    
    Disclaimer:
    This confidence score is an internal analytical metric based on spatial continuity, temporal continuity,
    and intensity consistency. It is NOT a calibrated statistical probability.
    """

    @classmethod
    def calculate_confidence(cls, track: WeatherEventTrack) -> float:
        if not track.points or len(track.points) == 1:
            return 0.50

        num_points = len(track.points)

        # 1. Length Bonus (up to 0.40)
        len_score = min(0.40, num_points * 0.10)

        # 2. Spatial Continuity Penalty
        displacements = []
        for i in range(1, num_points):
            p1 = track.points[i - 1]
            p2 = track.points[i]
            d = GeodesicMath.haversine_distance_km(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
            displacements.append(d)

        avg_disp = sum(displacements) / len(displacements) if displacements else 0.0
        # If avg jump > 200 km per step, penalize continuity
        spatial_score = max(0.0, 0.40 - (avg_disp / 500.0))

        # 3. Intensity Consistency
        intensities = [p.intensity for p in track.points]
        int_diffs = [abs(intensities[i] - intensities[i - 1]) for i in range(1, len(intensities))]
        avg_int_diff = sum(int_diffs) / len(int_diffs) if int_diffs else 0.0
        max_i = max(intensities) if max(intensities) > 0 else 1.0
        int_score = max(0.0, 0.20 - (avg_int_diff / (max_i + 1e-6)))

        total_conf = len_score + spatial_score + int_score
        return float(round(max(0.10, min(1.0, total_conf)), 3))
