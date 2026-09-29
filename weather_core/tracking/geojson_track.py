import json
from typing import List, Dict, Any
from .models import WeatherEventTrack


class GeoJSONTrackExporter:
    """
    Exports WeatherEventTrack objects to standard GeoJSON FeatureCollection containing
    LineString trajectory paths, Point centroids, and Polygon event regions.
    """

    @classmethod
    def tracks_to_geojson(cls, tracks: List[WeatherEventTrack]) -> Dict[str, Any]:
        features = []

        for trk in tracks:
            if not trk.points:
                continue

            # 1. LineString Trajectory Feature
            line_coords = [[p.longitude, p.latitude] for p in trk.points]
            track_feature = {
                "type": "Feature",
                "id": f"TRK_LINE_{trk.track_id}",
                "geometry": {
                    "type": "LineString",
                    "coordinates": line_coords,
                },
                "properties": {
                    "feature_type": "trajectory_path",
                    "track_id": trk.track_id,
                    "event_type": trk.event_type,
                    "start_time": trk.start_time,
                    "end_time": trk.end_time,
                    "total_displacement_km": trk.total_displacement_km,
                    "bearing_degrees": trk.direction_degrees,
                    "speed_kmh": trk.speed_kmh,
                    "intensity_change": trk.intensity_change,
                    "area_change": trk.area_change,
                    "confidence": trk.confidence,
                },
            }
            features.append(track_feature)

            # 2. Point Feature for Latest Position
            last_p = trk.points[-1]
            point_feature = {
                "type": "Feature",
                "id": f"TRK_POS_{trk.track_id}",
                "geometry": {
                    "type": "Point",
                    "coordinates": [last_p.longitude, last_p.latitude],
                },
                "properties": {
                    "feature_type": "current_position",
                    "track_id": trk.track_id,
                    "event_type": trk.event_type,
                    "severity": last_p.severity,
                    "timestamp": last_p.timestamp,
                    "peak_intensity": last_p.intensity,
                    "max_z_score": last_p.z_score,
                    "speed_kmh": trk.speed_kmh,
                    "bearing_degrees": trk.direction_degrees,
                },
            }
            features.append(point_feature)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "system": "SIH26078 AERIS Spatio-Temporal Tracking Engine",
                "total_tracks": len(tracks),
            },
            "features": features,
        }

    @classmethod
    def save_tracks_geojson(cls, tracks: List[WeatherEventTrack], file_path: str) -> str:
        data = cls.tracks_to_geojson(tracks)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return file_path
