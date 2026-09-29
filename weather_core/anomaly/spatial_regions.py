from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr

from .event_types import ExtremeEventType
from .severity import SeverityLevel, SeverityEngine


@dataclass
class CentroidCoordinates:
    latitude: float
    longitude: float

    def to_dict(self) -> Dict[str, float]:
        return {"latitude": self.latitude, "longitude": self.longitude}


@dataclass
class BoundingBox:
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
class ExtremeWeatherEvent:
    event_id: str
    event_type: str
    severity: str
    timestamp: str
    variable_name: str
    units: str
    centroid: CentroidCoordinates
    bounding_box: BoundingBox
    area_km2: float
    max_intensity: float
    mean_intensity: float
    max_z_score: float
    mean_z_score: float
    affected_grid_cells: int
    source_dataset: str
    provenance_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["centroid"] = self.centroid.to_dict()
        data["bounding_box"] = self.bounding_box.to_dict()
        return data


class SpatialRegionDetector:
    """
    Identifies connected spatial regions of extreme anomalous grid cells
    and extracts geometric/statistical attributes.
    """

    @classmethod
    def detect_connected_regions(
        self,
        z_score_da: xr.DataArray,
        val_da: xr.DataArray,
        clim_mean_da: xr.DataArray,
        var_name: str,
        timestamp: str,
        z_threshold: float = 2.0,
        source_dataset: str = "LocalNetCDF",
        min_cluster_cells: int = 2,
    ) -> List[ExtremeWeatherEvent]:
        """
        Extracts spatial clusters of anomalous cells (Z >= z_threshold) and builds Event objects.
        """
        z_vals = z_score_da.values
        val_vals = val_da.values
        lats = z_score_da.coords["latitude"].values
        lons = z_score_da.coords["longitude"].values

        # Binary mask of extreme anomalous cells
        anomaly_mask = (abs(z_vals) >= z_threshold) & (~np.isnan(z_vals))
        if not np.any(anomaly_mask):
            return []

        # 8-connectivity structure
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = ndimage.label(anomaly_mask, structure=structure)

        events = []
        severity_engine = SeverityEngine()

        lat_step = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.5
        lon_step = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.5
        # Cell area approximation in km^2
        cell_area_km2 = (lat_step * 111.0) * (lon_step * 111.0 * np.cos(np.radians(np.mean(lats))))

        for region_id in range(1, num_features + 1):
            region_indices = np.argwhere(labeled_array == region_id)
            if len(region_indices) < min_cluster_cells:
                continue

            r_lats = lats[region_indices[:, 0]]
            r_lons = lons[region_indices[:, 1]]
            r_z_vals = z_vals[region_indices[:, 0], region_indices[:, 1]]
            r_val_vals = val_vals[region_indices[:, 0], region_indices[:, 1]]

            # Weighted centroid calculation
            weights = np.abs(r_z_vals)
            weight_sum = np.sum(weights) if np.sum(weights) > 0 else 1.0
            centroid_lat = float(np.sum(r_lats * weights) / weight_sum)
            centroid_lon = float(np.sum(r_lons * weights) / weight_sum)

            min_lat = float(np.min(r_lats))
            max_lat = float(np.max(r_lats))
            min_lon = float(np.min(r_lons))
            max_lon = float(np.max(r_lons))

            max_z = float(np.max(r_z_vals)) if np.max(r_z_vals) > 0 else float(np.min(r_z_vals))
            mean_z = float(np.mean(r_z_vals))
            max_val = float(np.max(r_val_vals))
            mean_val = float(np.mean(r_val_vals))

            event_type = ExtremeEventType.from_variable(var_name, z_score=max_z).value
            severity = severity_engine.classify_z_score(max_z).value

            event_id = f"EVT_{var_name.upper()}_{timestamp.replace(':', '').replace('-', '')[:15]}_{region_id:03d}"

            event = ExtremeWeatherEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=timestamp,
                variable_name=var_name,
                units=str(val_da.attrs.get("units", "")),
                centroid=CentroidCoordinates(latitude=round(centroid_lat, 4), longitude=round(centroid_lon, 4)),
                bounding_box=BoundingBox(
                    min_lat=round(min_lat, 4),
                    max_lat=round(max_lat, 4),
                    min_lon=round(min_lon, 4),
                    max_lon=round(max_lon, 4),
                ),
                area_km2=round(float(len(region_indices) * cell_area_km2), 2),
                max_intensity=round(max_val, 4),
                mean_intensity=round(mean_val, 4),
                max_z_score=round(max_z, 4),
                mean_z_score=round(mean_z, 4),
                affected_grid_cells=int(len(region_indices)),
                source_dataset=source_dataset,
                provenance_metadata={
                    "z_threshold": z_threshold,
                    "resolution_deg": round(lat_step, 4),
                    "calculation_method": "ZScoreDetector + 8-Connected Component Region Detection",
                },
            )
            events.append(event)

        return events
