from typing import Dict, Any, Tuple
import numpy as np
import xarray as xr


class QualityControlError(Exception):
    """Raised when quality control validation fails or critical physical violations occur."""
    pass


class QualityControlChecker:
    """
    Meteorological Quality Control (QC) Checker.
    Enforces documented, physically defensible bounds for weather variables.
    
    Documented Physical Thresholds:
    1. Total Precipitation: [0.0, 1000.0] mm (Precipitation must be non-negative)
    2. Air Temperature: [180.0, 340.0] K (Surface temperature range on Earth)
    3. Wind Speed: [0.0, 150.0] m/s (Wind speed must be non-negative)
    4. Surface Pressure: [300.0, 1100.0] hPa or [30000.0, 110000.0] Pa
    """

    PHYSICAL_BOUNDS = {
        "precipitation": {
            "min": 0.0,
            "max": 1000.0,
            "units": "mm",
            "description": "Precipitation must be non-negative (>= 0 mm) and below world record hourly precip."
        },
        "temperature": {
            "min": 180.0,  # ~ -93.2°C (Vostok Station record low)
            "max": 340.0,  # ~ 66.8°C (Death Valley record high ~ 56.7°C + margin)
            "units": "K",
            "description": "Earth surface temperature bound in Kelvin."
        },
        "wind_speed": {
            "min": 0.0,
            "max": 150.0,  # ~ 540 km/h (Category 5 Cyclone / Tornado max winds)
            "units": "m/s",
            "description": "Wind speed magnitude must be non-negative (>= 0 m/s)."
        },
        "pressure": {
            "min": 30000.0,  # Pa (300 hPa upper atmospheric surface pressure)
            "max": 110000.0, # Pa (1100 hPa high pressure record)
            "units": "Pa",
            "description": "Surface pressure bound in Pascals."
        }
    }

    @classmethod
    def get_var_type(cls, var_name: str) -> str:
        name_lower = var_name.lower()
        if "precip" in name_lower or name_lower in ("tp", "pr"):
            return "precipitation"
        elif "temp" in name_lower or name_lower in ("t2m", "t", "t_2m"):
            return "temperature"
        elif "wind" in name_lower or name_lower in ("ws10", "si10", "u10", "v10"):
            return "wind_speed"
        elif "press" in name_lower or name_lower in ("sp", "msl", "p"):
            return "pressure"
        return "unknown"

    @classmethod
    def run_qc(cls, data_array: xr.DataArray, var_name: str) -> Tuple[xr.DataArray, Dict[str, Any]]:
        """
        Runs quality control on a DataArray.
        Returns: (qc_flag_mask, summary_dict)
        QC Flags:
          0 = Valid value
          1 = Out of physical bounds
          2 = Missing / NaN value
        """
        vals = data_array.values
        var_type = cls.get_var_type(var_name)
        bounds = cls.PHYSICAL_BOUNDS.get(var_type, None)

        qc_flags = np.zeros(vals.shape, dtype=np.int8)

        # Flag missing values (Flag 2)
        missing_mask = np.isnan(vals) if np.issubdtype(vals.dtype, np.inexact) else np.zeros(vals.shape, dtype=bool)
        qc_flags[missing_mask] = 2

        out_of_bounds_count = 0
        min_viol = None
        max_viol = None

        if bounds:
            min_bound = bounds["min"]
            max_bound = bounds["max"]

            # If temperature is in Celsius [-93, 60], adjust bounds check
            units = str(data_array.attrs.get("units", "")).strip()
            if var_type == "temperature" and units in ("degC", "C", "deg_C"):
                min_bound -= 273.15
                max_bound -= 273.15
            elif var_type == "pressure" and units in ("hPa", "mb"):
                min_bound /= 100.0
                max_bound /= 100.0

            valid_vals = vals[~missing_mask]
            if valid_vals.size > 0:
                oob_mask = (vals < min_bound) | (vals > max_bound)
                oob_mask = oob_mask & (~missing_mask)
                qc_flags[oob_mask] = 1
                out_of_bounds_count = int(np.sum(oob_mask))

                if out_of_bounds_count > 0:
                    violating_vals = vals[oob_mask]
                    min_viol = float(np.min(violating_vals))
                    max_viol = float(np.max(violating_vals))

        total_cells = vals.size
        valid_count = int(np.sum(qc_flags == 0))
        missing_count = int(np.sum(qc_flags == 2))

        summary = {
            "variable": var_name,
            "variable_type": var_type,
            "total_cells": total_cells,
            "valid_cells": valid_count,
            "out_of_bounds_cells": out_of_bounds_count,
            "missing_cells": missing_count,
            "qc_passed": out_of_bounds_count == 0,
            "applied_bounds": bounds,
            "min_violation": min_viol,
            "max_violation": max_viol,
        }

        qc_da = xr.DataArray(
            qc_flags,
            dims=data_array.dims,
            coords=data_array.coords,
            name=f"{var_name}_qc_flag",
            attrs={
                "long_name": f"Quality Control Flag for {var_name}",
                "flag_values": [0, 1, 2],
                "flag_meanings": "valid out_of_bounds missing",
            },
        )

        return qc_da, summary
