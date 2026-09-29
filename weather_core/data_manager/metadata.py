from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class GeographicalBounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }


@dataclass
class DatasetMetadata:
    """
    DatasetMetadata schema tracking dataset provenance, physical units, spatial bounds, and size.
    """
    dataset_name: str
    variable: str
    units: str
    source: str  # ERA5, IMDAA, NCUM, NEPS-G, LocalGRIB, LocalNetCDF
    geographical_bounds: GeographicalBounds
    time_start: str
    time_end: str
    resolution_km: float
    local_or_cloud: str  # "local" or "cloud"
    file_location: str
    size_bytes: int
    checksum: str = ""
    forecast_lead_hours: Optional[int] = None
    ensemble_member: Optional[str] = None
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["geographical_bounds"] = self.geographical_bounds.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        bounds_dict = data.get("geographical_bounds", {})
        bounds = GeographicalBounds(
            min_lat=float(bounds_dict.get("min_lat", 0.0)),
            max_lat=float(bounds_dict.get("max_lat", 0.0)),
            min_lon=float(bounds_dict.get("min_lon", 0.0)),
            max_lon=float(bounds_dict.get("max_lon", 0.0)),
        )
        return cls(
            dataset_name=str(data["dataset_name"]),
            variable=str(data["variable"]),
            units=str(data["units"]),
            source=str(data["source"]),
            geographical_bounds=bounds,
            time_start=str(data["time_start"]),
            time_end=str(data["time_end"]),
            resolution_km=float(data["resolution_km"]),
            local_or_cloud=str(data["local_or_cloud"]),
            file_location=str(data["file_location"]),
            size_bytes=int(data.get("size_bytes", 0)),
            checksum=str(data.get("checksum", "")),
            forecast_lead_hours=data.get("forecast_lead_hours"),
            ensemble_member=data.get("ensemble_member"),
            extra_attributes=data.get("extra_attributes", {}),
        )
