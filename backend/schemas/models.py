from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class CentroidSchema(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Centroid latitude in degrees North")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Centroid longitude in degrees East")


class BoundingBoxSchema(BaseModel):
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)
    min_lon: float = Field(..., ge=-180.0, le=180.0)
    max_lon: float = Field(..., ge=-180.0, le=180.0)


class WeatherEventSchema(BaseModel):
    event_id: str
    event_type: str
    severity: str
    timestamp: str
    variable_name: str
    units: str
    centroid: CentroidSchema
    bounding_box: BoundingBoxSchema
    area_km2: float = Field(..., ge=0.0)
    max_intensity: float
    mean_intensity: float
    max_z_score: float
    mean_z_score: float
    affected_grid_cells: int = Field(..., ge=1)
    source_dataset: str
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)


class TrajectoryPointSchema(BaseModel):
    timestamp: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    intensity: float
    area_km2: float
    severity: str
    z_score: float
    event_id: str


class WeatherEventTrackSchema(BaseModel):
    track_id: str
    event_type: str
    severity: str
    start_time: str
    end_time: str
    current_position: CentroidSchema
    direction_degrees: float
    speed_kmh: float
    intensity_change: float
    area_change: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    trajectory: List[TrajectoryPointSchema]


class AlertSchema(BaseModel):
    alert_id: str
    event_id: str
    event_type: str
    severity: str
    latitude: float
    longitude: float
    timestamp: str
    anomaly_score: float
    affected_area_km2: float
    confidence: float
    source_dataset: str
    disclaimer: str = "AERIS internal analytical alert — not an official government warning."


class DatasetMetadataSchema(BaseModel):
    dataset_name: str
    variable: str
    units: str
    source: str
    geographical_bounds: BoundingBoxSchema
    time_start: str
    time_end: str
    resolution_km: float
    local_or_cloud: str
    file_location: str
    size_bytes: int
    processing_version: str = "0.1.0-dev"


class GeoJSONFeatureSchema(BaseModel):
    type: str = "Feature"
    id: str
    geometry: Dict[str, Any]
    properties: Dict[str, Any]


class GeoJSONFeatureCollectionSchema(BaseModel):
    type: str = "FeatureCollection"
    metadata: Dict[str, Any]
    features: List[GeoJSONFeatureSchema]
