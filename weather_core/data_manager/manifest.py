from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import yaml


@dataclass
class DatasetManifest:
    """
    DatasetManifest schema defining dataset provenance, licensing, spatial/temporal scope,
    and storage paths for AERIS multi-timestamp meteorological datasets.
    """
    dataset_name: str
    source: str  # ERA5, IMDAA, NCUM, ECMWF Open Data, NOAA GFS
    source_url: str
    variables: List[str]
    units: Dict[str, str]
    region: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    resolution: str
    time_range: Dict[str, str]  # start_time, end_time
    download_size_mb: float
    license_notes: str
    local_path: str
    dataset_type: str = "OBSERVED/REANALYSIS"  # "OBSERVED/REANALYSIS" or "NWP FORECAST"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_name=str(data["dataset_name"]),
            source=str(data["source"]),
            source_url=str(data["source_url"]),
            variables=list(data.get("variables", [])),
            units=dict(data.get("units", {})),
            region=dict(data.get("region", {})),
            resolution=str(data.get("resolution", "")),
            time_range=dict(data.get("time_range", {})),
            download_size_mb=float(data.get("download_size_mb", 0.0)),
            license_notes=str(data.get("license_notes", "")),
            local_path=str(data.get("local_path", "")),
            dataset_type=str(data.get("dataset_type", "OBSERVED/REANALYSIS")),
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "DatasetManifest":
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
