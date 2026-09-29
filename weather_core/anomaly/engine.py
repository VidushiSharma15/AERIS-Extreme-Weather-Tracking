from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import xarray as xr

from .thresholds import ThresholdConfig
from .spatial_regions import SpatialRegionDetector, ExtremeWeatherEvent
from .severity import SeverityEngine
from weather_core.climatology import ClimatologyEngine
from weather_core.config import get_settings


class AnomalyEngine:
    """
    Core AnomalyEngine for SIH26078 AERIS.
    Calculates absolute anomalies, Z-scores, percentile anomalies, connected spatial regions,
    and classifies event severities.
    """

    EPSILON = 1e-6  # Safe guard against divide-by-zero for near-zero standard deviation

    def __init__(
        self,
        climatology_baseline: Optional[xr.Dataset] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ):
        self.climatology_baseline = climatology_baseline
        self.threshold_config = threshold_config or ThresholdConfig()
        self.settings = get_settings()

    def compute_anomaly_fields(
        self,
        ds: xr.Dataset,
        climatology_baseline: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """
        Calculates absolute anomaly and standardized Z-score for all physical data variables.
        """
        clim_ds = climatology_baseline if climatology_baseline is not None else self.climatology_baseline
        if clim_ds is None:
            # Fallback: compute baseline from input dataset itself
            clim_engine = ClimatologyEngine(historical_dataset=ds)
            clim_ds = clim_engine.compute_baseline_statistics()

        ds_anomaly = ds.copy()

        for var_name in list(ds.data_vars.keys()):
            if str(var_name).endswith("_qc_flag") or str(var_name).endswith("_anomaly") or str(var_name).endswith("_zscore"):
                continue

            mean_key = f"{var_name}_clim_mean"
            std_key = f"{var_name}_clim_std"

            if mean_key in clim_ds and std_key in clim_ds:
                clim_mean = clim_ds[mean_key]
                clim_std = clim_ds[std_key]
            else:
                clim_mean = ds[var_name].mean(dim="time", skipna=True) if "time" in ds[var_name].dims else ds[var_name]
                clim_std = ds[var_name].std(dim="time", skipna=True) if "time" in ds[var_name].dims else xr.zeros_like(ds[var_name])

            # 1. Absolute Anomaly
            abs_anomaly = ds[var_name] - clim_mean
            ds_anomaly[f"{var_name}_abs_anomaly"] = abs_anomaly
            ds_anomaly[f"{var_name}_abs_anomaly"].attrs.update({
                "long_name": f"Absolute Anomaly for {var_name}",
                "units": str(ds[var_name].attrs.get("units", "")),
            })

            # 2. Standardized Z-Score Anomaly (Safe Division)
            safe_std = xr.where(clim_std == 0.0, self.EPSILON, clim_std)
            z_score = abs_anomaly / safe_std
            ds_anomaly[f"{var_name}_zscore"] = z_score
            ds_anomaly[f"{var_name}_zscore"].attrs.update({
                "long_name": f"Standardized Z-Score Anomaly for {var_name}",
                "units": "dimensionless_zscore",
            })

        ds_anomaly.attrs["climatology_provenance"] = str(clim_ds.attrs.get("climatology_status", "Calculated"))
        return ds_anomaly

    def detect_extreme_events(
        self,
        ds: xr.Dataset,
        climatology_baseline: Optional[xr.Dataset] = None,
        source_dataset_name: str = "LocalNetCDF",
    ) -> Tuple[xr.Dataset, List[ExtremeWeatherEvent]]:
        """
        Runs anomaly field computation and extracts connected spatial extreme event regions.
        """
        ds_anomaly = self.compute_anomaly_fields(ds, climatology_baseline=climatology_baseline)
        clim_ds = climatology_baseline if climatology_baseline is not None else self.climatology_baseline
        if clim_ds is None:
            clim_engine = ClimatologyEngine(historical_dataset=ds)
            clim_ds = clim_engine.compute_baseline_statistics()

        all_events: List[ExtremeWeatherEvent] = []

        time_steps = len(ds["time"]) if "time" in ds.dims else 1

        for t_idx in range(time_steps):
            t_ds = ds.isel(time=t_idx) if "time" in ds.dims else ds
            t_anom = ds_anomaly.isel(time=t_idx) if "time" in ds_anomaly.dims else ds_anomaly
            t_stamp = str(t_ds["time"].values) if "time" in t_ds.coords else "2026-09-25T00:00:00"

            for var_name in list(ds.data_vars.keys()):
                if str(var_name).endswith("_qc_flag") or str(var_name).endswith("_anomaly") or str(var_name).endswith("_zscore"):
                    continue

                z_key = f"{var_name}_zscore"
                if z_key not in t_anom:
                    continue

                threshold_cfg = self.threshold_config.get(var_name)
                z_thresh = threshold_cfg.z_score_threshold

                val_da = t_ds[var_name]
                z_score_da = t_anom[z_key]
                mean_key = f"{var_name}_clim_mean"
                clim_mean_da = clim_ds[mean_key] if mean_key in clim_ds else val_da

                events = SpatialRegionDetector.detect_connected_regions(
                    z_score_da=z_score_da,
                    val_da=val_da,
                    clim_mean_da=clim_mean_da,
                    var_name=var_name,
                    timestamp=t_stamp,
                    z_threshold=z_thresh,
                    source_dataset=source_dataset_name,
                )
                all_events.extend(events)

        return ds_anomaly, all_events
