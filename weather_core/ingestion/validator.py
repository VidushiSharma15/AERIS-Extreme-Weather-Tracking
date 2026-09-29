from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import xarray as xr


class DatasetValidationError(Exception):
    """Raised when a weather dataset fails validation rules."""
    pass


class DatasetValidator:
    """
    Validates meteorological datasets for coordinate completeness, physical unit standardity,
    data integrity, and spatial dimension consistency.
    """

    ALLOWED_LAT_NAMES = {"latitude", "lat", "NAV_LAT", "y"}
    ALLOWED_LON_NAMES = {"longitude", "lon", "NAV_LON", "x"}
    ALLOWED_TIME_NAMES = {"time", "valid_time", "step", "t"}

    SUPPORTED_UNITS = {
        "precipitation": {"mm", "m", "kg m-2", "kg m-2 s-1", "mm/hr", "mm/day"},
        "temperature": {"K", "degC", "C", "deg_C"},
        "wind_speed": {"m/s", "m s-1", "kt", "km/h"},
    }

    VARIABLE_MAPPINGS = {
        "total_precipitation": "precipitation",
        "tp": "precipitation",
        "precip": "precipitation",
        "precipitation": "precipitation",
        "2m_temperature": "temperature",
        "t2m": "temperature",
        "temp": "temperature",
        "temperature": "temperature",
        "10m_wind_speed": "wind_speed",
        "ws10": "wind_speed",
        "wind_speed": "wind_speed",
        "si10": "wind_speed",
    }

    @classmethod
    def find_coordinate(cls, ds: xr.Dataset, allowed_names: set) -> Optional[str]:
        for name in ds.coords:
            if str(name).lower() in allowed_names or name in allowed_names:
                return str(name)
        for name in ds.dims:
            if str(name).lower() in allowed_names or name in allowed_names:
                return str(name)
        return None

    @classmethod
    def validate_dataset(cls, ds: xr.Dataset) -> Dict[str, Any]:
        """
        Validates an xarray Dataset and returns dataset health metrics.
        Raises DatasetValidationError if required coordinates or structure are missing.
        """
        if ds is None or not isinstance(ds, xr.Dataset):
            raise DatasetValidationError("Corrupted or empty dataset object.")

        if len(ds.data_vars) == 0:
            raise DatasetValidationError("Dataset contains no data variables.")

        lat_name = cls.find_coordinate(ds, cls.ALLOWED_LAT_NAMES)
        lon_name = cls.find_coordinate(ds, cls.ALLOWED_LON_NAMES)
        time_name = cls.find_coordinate(ds, cls.ALLOWED_TIME_NAMES)

        missing_coords = []
        if not lat_name:
            missing_coords.append("latitude")
        if not lon_name:
            missing_coords.append("longitude")
        if not time_name:
            missing_coords.append("time")

        if missing_coords:
            raise DatasetValidationError(
                f"Dataset is missing mandatory coordinates: {', '.join(missing_coords)}. "
                f"Available coords/dims: {list(ds.coords.keys())} / {list(ds.dims.keys())}"
            )

        # Coordinate value checks
        lats = ds[lat_name].values
        lons = ds[lon_name].values

        if np.any(np.isnan(lats)) or np.any(np.isnan(lons)):
            raise DatasetValidationError("Latitude or Longitude coordinates contain NaN values.")

        if np.min(lats) < -90.0 or np.max(lats) > 90.0:
            raise DatasetValidationError(f"Latitude range out of physical bounds [-90, 90]: [{np.min(lats)}, {np.max(lats)}]")

        # Validate variables and units
        var_summaries = {}
        for var_name, data_array in ds.data_vars.items():
            units = str(data_array.attrs.get("units", "unknown")).strip()
            var_type = cls.VARIABLE_MAPPINGS.get(str(var_name).lower(), None)

            if var_type and units != "unknown":
                allowed = cls.SUPPORTED_UNITS.get(var_type, set())
                if units not in allowed and not any(u.lower() == units.lower() for u in allowed):
                    raise DatasetValidationError(
                        f"Variable '{var_name}' has invalid or unsupported unit '{units}'. "
                        f"Expected one of: {allowed}"
                    )

            vals = data_array.values
            total_elements = vals.size
            nan_count = int(np.isnan(vals).sum()) if np.issubdtype(vals.dtype, np.inexact) else 0
            missing_pct = (nan_count / total_elements * 100.0) if total_elements > 0 else 0.0

            valid_vals = vals[~np.isnan(vals)] if np.issubdtype(vals.dtype, np.inexact) else vals
            min_val = float(np.min(valid_vals)) if valid_vals.size > 0 else float("nan")
            max_val = float(np.max(valid_vals)) if valid_vals.size > 0 else float("nan")
            mean_val = float(np.mean(valid_vals)) if valid_vals.size > 0 else float("nan")

            var_summaries[str(var_name)] = {
                "units": units,
                "shape": list(data_array.shape),
                "dims": [str(d) for d in data_array.dims],
                "total_elements": total_elements,
                "missing_count": nan_count,
                "missing_pct": round(missing_pct, 2),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "mean": round(mean_val, 4),
            }

        # Spatial resolution estimate
        lat_step = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.0
        lon_step = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.0
        approx_res_deg = (lat_step + lon_step) / 2.0
        approx_res_km = approx_res_deg * 111.0  # ~111 km per degree

        return {
            "valid": True,
            "latitude_coord": lat_name,
            "longitude_coord": lon_name,
            "time_coord": time_name,
            "lat_range": [float(np.min(lats)), float(np.max(lats))],
            "lon_range": [float(np.min(lons)), float(np.max(lons))],
            "time_range": [str(ds[time_name].values[0]), str(ds[time_name].values[-1])],
            "time_steps": int(len(ds[time_name])),
            "resolution_deg": round(approx_res_deg, 4),
            "resolution_km": round(approx_res_km, 2),
            "variables": var_summaries,
        }
