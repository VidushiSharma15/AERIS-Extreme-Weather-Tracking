from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr


class ClimatologyEngine:
    """
    ClimatologyEngine calculates grid-cell level historical climatological baselines
    (mean, standard deviation, median, percentiles, quantiles).
    
    Scientific Honesty Note:
    Operational climatological baselines require 30 years of historical reanalysis (e.g. ERA5 1991-2020).
    If the provided historical dataset has less than 30 years of data, the engine explicitly
    flags this condition in its output metadata.
    """

    MIN_OPERATIONAL_YEARS = 30

    def __init__(self, historical_dataset: Optional[xr.Dataset] = None):
        self.dataset = historical_dataset

    def compute_baseline_statistics(
        self,
        ds: Optional[xr.Dataset] = None,
        percentiles: Tuple[float, ...] = (90.0, 95.0, 99.0),
    ) -> xr.Dataset:
        target_ds = ds if ds is not None else self.dataset
        if target_ds is None:
            raise ValueError("No historical dataset provided for climatology baseline computation.")

        if "time" not in target_ds.dims and "time" not in target_ds.coords:
            raise ValueError("Dataset missing 'time' dimension required for climatology calculation.")

        time_years = 1
        if "time" in target_ds.coords and np.issubdtype(target_ds["time"].dtype, np.datetime64):
            try:
                time_years = len(np.unique(target_ds["time"].dt.year.values))
            except Exception:
                time_years = 1

        is_sufficient = time_years >= self.MIN_OPERATIONAL_YEARS

        status_msg = (
            "Operational 30-year climatology baseline."
            if is_sufficient
            else f"Development dataset is insufficient for operational 30-year climatology (Contains {time_years} year(s))."
        )

        baseline_vars = {}
        for var_name, da in target_ds.data_vars.items():
            if str(var_name).endswith("_qc_flag") or str(var_name).endswith("_anomaly") or str(var_name).endswith("_zscore"):
                continue

            mean_da = da.mean(dim="time", skipna=True)
            std_da = da.std(dim="time", skipna=True)
            median_da = da.median(dim="time", skipna=True)

            baseline_vars[f"{var_name}_clim_mean"] = mean_da
            baseline_vars[f"{var_name}_clim_std"] = std_da
            baseline_vars[f"{var_name}_clim_median"] = median_da

            for p in percentiles:
                p_val = float(p)
                p_da = da.quantile(p_val / 100.0, dim="time", skipna=True)
                if "quantile" in p_da.coords:
                    p_da = p_da.drop_vars("quantile")
                baseline_vars[f"{var_name}_clim_p{int(p_val)}"] = p_da

        baseline_ds = xr.Dataset(data_vars=baseline_vars)
        baseline_ds.attrs["climatology_status"] = status_msg
        baseline_ds.attrs["historical_time_years"] = time_years
        baseline_ds.attrs["is_operational_30yr"] = is_sufficient
        return baseline_ds
