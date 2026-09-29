from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import xarray as xr


class StandardizationError(Exception):
    """Raised when dataset standardization fails."""
    pass


class CoordinateStandardizer:
    """
    Standardizes spatial coordinates (latitude, longitude) across diverse NWP/Reanalysis formats.
    """

    LAT_RENAMES = {"lat": "latitude", "NAV_LAT": "latitude", "y": "latitude"}
    LON_RENAMES = {"lon": "longitude", "NAV_LON": "longitude", "x": "longitude"}

    @classmethod
    def standardize_coordinates(cls, ds: xr.Dataset, normalize_lon_180: bool = False) -> xr.Dataset:
        """
        Renames coordinates to standard 'latitude' and 'longitude',
        ensures monotonic ascending order, and detects irregular grids.
        """
        ds_out = ds.copy()

        # Rename lat/lon if needed
        rename_map = {}
        for k in list(ds_out.coords.keys()) + list(ds_out.dims):
            k_str = str(k)
            if k_str in cls.LAT_RENAMES and k_str != "latitude":
                rename_map[k] = "latitude"
            elif k_str in cls.LON_RENAMES and k_str != "longitude":
                rename_map[k] = "longitude"

        if rename_map:
            ds_out = ds_out.rename(rename_map)

        if "latitude" not in ds_out.coords or "longitude" not in ds_out.coords:
            raise StandardizationError("Dataset missing standardized 'latitude' or 'longitude' coordinates.")

        lats = ds_out["latitude"].values
        lons = ds_out["longitude"].values

        # Check for irregular / 2D curvilinear grids
        if lats.ndim > 1 or lons.ndim > 1:
            ds_out.attrs["grid_type"] = "irregular_curvilinear"
        else:
            ds_out.attrs["grid_type"] = "regular_rectilinear"

        # Normalize longitude from [0, 360] to [-180, 180] if requested
        if normalize_lon_180 and np.any(lons > 180.0):
            lons_norm = np.where(lons > 180.0, lons - 360.0, lons)
            ds_out = ds_out.assign_coords(longitude=lons_norm)

        # Sort coordinates monotonically ascending
        ds_out = ds_out.sortby("latitude")
        ds_out = ds_out.sortby("longitude")

        # Standard CF Coordinate Metadata
        ds_out["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})
        ds_out["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

        return ds_out


class TimeStandardizer:
    """
    Standardizes temporal dimensions, ensures datetime64 UTC formatting,
    chronological sorting, and duplicate handling.
    """

    TIME_RENAMES = {"valid_time": "time", "step": "time", "t": "time"}

    @classmethod
    def standardize_time(cls, ds: xr.Dataset) -> xr.Dataset:
        ds_out = ds.copy()

        rename_map = {}
        for k in list(ds_out.coords.keys()) + list(ds_out.dims):
            k_str = str(k)
            if k_str in cls.TIME_RENAMES and k_str != "time":
                rename_map[k] = "time"

        if rename_map:
            ds_out = ds_out.rename(rename_map)

        if "time" not in ds_out.coords:
            raise StandardizationError("Dataset missing standardized 'time' coordinate.")

        # Convert to datetime64 if not already
        times = ds_out["time"].values
        if not np.issubdtype(times.dtype, np.datetime64):
            try:
                times_converted = pd.to_datetime(times)
                ds_out = ds_out.assign_coords(time=times_converted)
            except Exception as e:
                raise StandardizationError(f"Failed to parse time coordinates into datetime64: {str(e)}")

        # Deduplicate timestamps if duplicates exist
        unique_times, counts = np.unique(ds_out["time"].values, return_counts=True)
        if np.any(counts > 1):
            ds_out = ds_out.drop_duplicates(dim="time")
            ds_out.attrs["duplicate_timestamps_found"] = True

        # Sort chronologically ascending
        ds_out = ds_out.sortby("time")
        ds_out["time"].attrs.update({"standard_name": "time"})

        return ds_out


class UnitStandardizer:
    """
    Handles explicit, safe meteorological unit conversions.
    """

    @classmethod
    def convert_temperature(cls, da: xr.DataArray, target_unit: str = "degC") -> xr.DataArray:
        current_unit = str(da.attrs.get("units", "")).strip()
        da_out = da.copy()

        if current_unit in ("K", "Kelvin") and target_unit in ("degC", "C", "deg_C"):
            da_out = da_out - 273.15
            da_out.attrs["units"] = "degC"
            da_out.attrs["unit_conversion_applied"] = f"K to degC (-273.15)"
        elif current_unit in ("degC", "C", "deg_C") and target_unit == "K":
            da_out = da_out + 273.15
            da_out.attrs["units"] = "K"
            da_out.attrs["unit_conversion_applied"] = f"degC to K (+273.15)"
        return da_out

    @classmethod
    def convert_precipitation(cls, da: xr.DataArray, target_unit: str = "mm") -> xr.DataArray:
        current_unit = str(da.attrs.get("units", "")).strip()
        da_out = da.copy()

        if current_unit == "m" and target_unit == "mm":
            da_out = da_out * 1000.0
            da_out.attrs["units"] = "mm"
            da_out.attrs["unit_conversion_applied"] = "m to mm (*1000.0)"
        return da_out


class MissingDataHandler:
    """
    Handles missing weather data safely without silent zero-fills.
    """

    @classmethod
    def analyze_missing(cls, ds: xr.Dataset) -> Dict[str, Any]:
        results = {}
        for var_name, da in ds.data_vars.items():
            vals = da.values
            total = vals.size
            nans = int(np.isnan(vals).sum()) if np.issubdtype(vals.dtype, np.inexact) else 0
            results[str(var_name)] = {
                "total_count": total,
                "missing_count": nans,
                "missing_pct": round((nans / total * 100.0) if total > 0 else 0.0, 2),
            }
        return results

    @classmethod
    def fill_missing_spatial_interpolation(cls, da: xr.DataArray, max_missing_pct: float = 5.0) -> xr.DataArray:
        """
        Interpolates small missing gaps using 2D spatial linear/nearest interpolation.
        Raises StandardizationError if missing percentage exceeds max_missing_pct.
        """
        vals = da.values
        nans = np.isnan(vals).sum() if np.issubdtype(vals.dtype, np.inexact) else 0
        pct = (nans / vals.size * 100.0) if vals.size > 0 else 0.0

        if pct == 0.0:
            return da

        if pct > max_missing_pct:
            raise StandardizationError(
                f"Missing data ratio ({pct:.2f}%) exceeds maximum safe interpolation threshold ({max_missing_pct}%). "
                f"Cannot safely interpolate array."
            )

        # Apply xarray linear interpolation along spatial dimensions if present
        da_filled = da.copy()
        if "latitude" in da.dims and "longitude" in da.dims:
            da_filled = da_filled.interpolate_na(dim="longitude", method="linear", fill_value="extrapolate")
            da_filled = da_filled.interpolate_na(dim="latitude", method="linear", fill_value="extrapolate")

        da_filled.attrs["missing_data_interpolated"] = True
        da_filled.attrs["missing_data_original_pct"] = round(pct, 2)
        return da_filled
