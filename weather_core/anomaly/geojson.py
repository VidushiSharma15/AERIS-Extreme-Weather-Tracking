import json
from typing import List, Dict, Any
from .spatial_regions import ExtremeWeatherEvent


class GeoJSONExporter:
    """
    Exports detected ExtremeWeatherEvent regions to standard GeoJSON FeatureCollection format.
    """

    @classmethod
    def events_to_geojson(cls, events: List[ExtremeWeatherEvent]) -> Dict[str, Any]:
        features = []
        for event in events:
            bbox = event.bounding_box
            # Build bounding box polygon coordinates [lon, lat]
            polygon_coords = [
                [
                    [bbox.min_lon, bbox.min_lat],
                    [bbox.max_lon, bbox.min_lat],
                    [bbox.max_lon, bbox.max_lat],
                    [bbox.min_lon, bbox.max_lat],
                    [bbox.min_lon, bbox.min_lat],  # Closed ring
                ]
            ]

            feature = {
                "type": "Feature",
                "id": event.event_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": polygon_coords,
                },
                "properties": {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "timestamp": event.timestamp,
                    "variable": event.variable_name,
                    "units": event.units,
                    "centroid": event.centroid.to_dict(),
                    "area_km2": event.area_km2,
                    "max_intensity": event.max_intensity,
                    "mean_intensity": event.mean_intensity,
                    "max_z_score": event.max_z_score,
                    "affected_grid_cells": event.affected_grid_cells,
                    "source_dataset": event.source_dataset,
                    "provenance": event.provenance_metadata,
                },
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "system": "SIH26078 AERIS Anomaly Detection Engine",
                "total_events": len(events),
            },
            "features": features,
        }

    @classmethod
    def save_geojson(cls, events: List[ExtremeWeatherEvent], file_path: str) -> str:
        data = cls.events_to_geojson(events)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return file_path
