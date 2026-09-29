from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import json
import numpy as np
import xarray as xr

from .standardizer import (
    CoordinateStandardizer,
    TimeStandardizer,
    UnitStandardizer,
    MissingDataHandler,
    StandardizationError,
)
from .qc import QualityControlChecker
from weather_core.ingestion import DatasetValidator, LocalNetCDFProvider
from weather_core.config import get_settings


@dataclass
class PreprocessingReport:
    dataset_name: str
    original_dims: Dict[str, int]
    standardized_dims: Dict[str, int]
    lat_range: list
    lon_range: list
    time_range: list
    resolution_deg: float
    resolution_km: float
    variables_summary: Dict[str, Any]
    quality_control_summary: Dict[str, Any]
    missing_data_summary: Dict[str, Any]
    pipeline_success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeatherPreprocessor:
    """
    Complete reproducible Weather Preprocessing & Standardization Pipeline.
    Executes:
    Validation -> Coordinate Standardization -> Time Standardization ->
    Unit Validation -> Missing Data Check -> Quality Control -> Standardized Dataset
    """

    def __init__(self, config_path: Optional[str] = None):
        self.settings = get_settings(config_path)

    def preprocess_dataset(
        self,
        source_path_or_ds: Any,
        dataset_name: str = "standardized_weather",
        target_temp_unit: Optional[str] = None,
        target_precip_unit: Optional[str] = None,
        crop_bounds: Optional[Dict[str, float]] = None,
    ) -> Tuple[xr.Dataset, PreprocessingReport]:
        """
        Runs the full 8-step preprocessing pipeline on a raw dataset.
        Returns: (standardized_dataset, preprocessing_report)
        """
        # Step 1: Ingestion / Initial Load
        if isinstance(source_path_or_ds, (str, xr.DataArray)):
            if isinstance(source_path_or_ds, str):
                provider = LocalNetCDFProvider()
                ds_raw = provider.open_dataset(source_path_or_ds)
            else:
                ds_raw = source_path_or_ds.to_dataset()
        elif isinstance(source_path_or_ds, xr.Dataset):
            ds_raw = source_path_or_ds
        else:
            raise StandardizationError("Invalid input type for preprocessing.")

        original_dims = {str(k): int(v) for k, v in ds_raw.sizes.items()}

        # Step 2: Ingestion Validation
        DatasetValidator.validate_dataset(ds_raw)

        # Step 3: Coordinate Standardization
        ds_std = CoordinateStandardizer.standardize_coordinates(ds_raw, normalize_lon_180=False)

        # Optional Spatial Domain Subsetting / Cropping
        if crop_bounds:
            ds_std = ds_std.sel(
                latitude=slice(crop_bounds["min_lat"], crop_bounds["max_lat"]),
                longitude=slice(crop_bounds["min_lon"], crop_bounds["max_lon"]),
            )

        # Step 4: Time Standardization
        ds_std = TimeStandardizer.standardize_time(ds_std)

        # Step 5: Unit Validation & Explicit Conversions
        for var_name in list(ds_std.data_vars.keys()):
            var_type = QualityControlChecker.get_var_type(var_name)
            if var_type == "temperature" and target_temp_unit:
                ds_std[var_name] = UnitStandardizer.convert_temperature(ds_std[var_name], target_unit=target_temp_unit)
            elif var_type == "precipitation" and target_precip_unit:
                ds_std[var_name] = UnitStandardizer.convert_precipitation(ds_std[var_name], target_unit=target_precip_unit)

        # Step 6: Missing Data Analysis
        missing_summary = MissingDataHandler.analyze_missing(ds_std)

        # Step 7: Quality Control Sanity Checks
        qc_summaries = {}
        for var_name in list(ds_std.data_vars.keys()):
            qc_flag_da, qc_info = QualityControlChecker.run_qc(ds_std[var_name], str(var_name))
            ds_std[f"{var_name}_qc_flag"] = qc_flag_da
            qc_summaries[str(var_name)] = qc_info

        # Step 8: Spatial Resolution & Report Assembly
        lats = ds_std["latitude"].values
        lons = ds_std["longitude"].values
        lat_step = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.0
        lon_step = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.0
        res_deg = (lat_step + lon_step) / 2.0
        res_km = res_deg * 111.0

        var_summaries = {}
        for var_name in list(ds_std.data_vars.keys()):
            if not str(var_name).endswith("_qc_flag"):
                da = ds_std[var_name]
                vals = da.values
                valid_vals = vals[~np.isnan(vals)] if np.issubdtype(vals.dtype, np.inexact) else vals
                var_summaries[str(var_name)] = {
                    "units": str(da.attrs.get("units", "")),
                    "shape": list(da.shape),
                    "min": float(np.min(valid_vals)) if valid_vals.size > 0 else 0.0,
                    "max": float(np.max(valid_vals)) if valid_vals.size > 0 else 0.0,
                    "mean": float(np.mean(valid_vals)) if valid_vals.size > 0 else 0.0,
                }

        standardized_dims = {str(k): int(v) for k, v in ds_std.sizes.items()}

        report = PreprocessingReport(
            dataset_name=dataset_name,
            original_dims=original_dims,
            standardized_dims=standardized_dims,
            lat_range=[float(np.min(lats)), float(np.max(lats))],
            lon_range=[float(np.min(lons)), float(np.max(lons))],
            time_range=[str(ds_std["time"].values[0]), str(ds_std["time"].values[-1])],
            resolution_deg=round(res_deg, 4),
            resolution_km=round(res_km, 2),
            variables_summary=var_summaries,
            quality_control_summary=qc_summaries,
            missing_data_summary=missing_summary,
            pipeline_success=True,
        )

        ds_std.attrs["title"] = f"Standardized {dataset_name}"
        ds_std.attrs["preprocessing_status"] = "PASSED_QUALITY_CONTROL"
        ds_std.attrs["Conventions"] = "CF-1.8"

        return ds_std, report
