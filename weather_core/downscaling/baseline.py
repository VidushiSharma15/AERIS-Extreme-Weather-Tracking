import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from typing import Dict, Any, Tuple, Optional
from .base import BaseDownscaler
from .metrics import AmplitudePreservationMetrics


class BilinearDownscaler(BaseDownscaler):
    """
    Bilinear / Bicubic spatial downscaler for meteorological grids.
    Refines coarse resolution fields (~28 km) to a localized 5 km target resolution grid.
    """

    def downscale_grid(
        self,
        ds: xr.Dataset,
        variable_name: str,
        target_resolution_km: Optional[float] = None,
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        target_res = target_resolution_km if target_resolution_km else self.target_resolution_km

        da_coarse = ds[variable_name]
        lats_coarse = ds.latitude.values
        lons_coarse = ds.longitude.values

        # Convert target resolution km to approximate degrees (1 deg ~ 111 km)
        ddeg = target_res / 111.0
        lats_fine = np.arange(lats_coarse.min(), lats_coarse.max() + ddeg / 2.0, ddeg)
        lons_fine = np.arange(lons_coarse.min(), lons_coarse.max() + ddeg / 2.0, ddeg)

        # Regrid using xarray interp (bilinear)
        da_fine = da_coarse.interp(latitude=lats_fine, longitude=lons_fine, method="linear")

        # Compute extreme amplitude metrics
        metrics = AmplitudePreservationMetrics.calculate(da_coarse.values, da_fine.values)

        meta = {
            "method": "Bilinear Spatial Refinement",
            "coarse_resolution_deg": float(abs(lats_coarse[1] - lats_coarse[0])) if len(lats_coarse) > 1 else 0.25,
            "target_resolution_km": target_res,
            "coarse_grid_shape": da_coarse.shape,
            "fine_grid_shape": da_fine.shape,
            "amplitude_metrics": metrics.to_dict(),
            "disclaimer": self.DISCLAIMER,
        }

        return da_fine, meta

    def downscale_event_region(
        self,
        ds: xr.Dataset,
        variable_name: str,
        bounding_box: Dict[str, float],
        target_resolution_km: Optional[float] = None,
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        target_res = target_resolution_km if target_resolution_km else self.target_resolution_km

        min_lat = bounding_box.get("min_lat", ds.latitude.values.min())
        max_lat = bounding_box.get("max_lat", ds.latitude.values.max())
        min_lon = bounding_box.get("min_lon", ds.longitude.values.min())
        max_lon = bounding_box.get("max_lon", ds.longitude.values.max())

        # Slice coarse dataset to event bounding box region
        ds_sub = ds.sel(
            latitude=slice(min_lat, max_lat),
            longitude=slice(min_lon, max_lon),
        )
        if len(ds_sub.latitude) == 0 or len(ds_sub.longitude) == 0:
            ds_sub = ds

        return self.downscale_grid(ds_sub, variable_name, target_resolution_km=target_res)


class StatisticalRefinementDownscaler(BaseDownscaler):
    """
    Statistical amplitude-preserving spatial refinement downscaler.
    Applies gradient-preserving spline interpolation with variance correction.
    """

    def downscale_grid(
        self,
        ds: xr.Dataset,
        variable_name: str,
        target_resolution_km: Optional[float] = None,
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        target_res = target_resolution_km if target_resolution_km else self.target_resolution_km
        da_coarse = ds[variable_name]

        lats_coarse = ds.latitude.values
        lons_coarse = ds.longitude.values
        ddeg = target_res / 111.0

        lats_fine = np.arange(lats_coarse.min(), lats_coarse.max() + ddeg / 2.0, ddeg)
        lons_fine = np.arange(lons_coarse.min(), lons_coarse.max() + ddeg / 2.0, ddeg)

        # Spline / cubic regridding
        da_fine = da_coarse.interp(latitude=lats_fine, longitude=lons_fine, method="cubic")

        # Amplitude correction to match coarse peak extreme precisely
        c_max = float(np.nanmax(da_coarse.values))
        f_max = float(np.nanmax(da_fine.values))
        if f_max > 0 and abs(f_max - c_max) > 1e-4:
            scale = c_max / f_max
            da_fine = da_fine * scale

        metrics = AmplitudePreservationMetrics.calculate(da_coarse.values, da_fine.values)

        meta = {
            "method": "Statistical Amplitude-Preserving Refinement",
            "target_resolution_km": target_res,
            "coarse_grid_shape": da_coarse.shape,
            "fine_grid_shape": da_fine.shape,
            "amplitude_metrics": metrics.to_dict(),
            "disclaimer": self.DISCLAIMER,
        }

        return da_fine, meta

    def downscale_event_region(
        self,
        ds: xr.Dataset,
        variable_name: str,
        bounding_box: Dict[str, float],
        target_resolution_km: Optional[float] = None,
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        target_res = target_resolution_km if target_resolution_km else self.target_resolution_km

        min_lat = bounding_box.get("min_lat", ds.latitude.values.min())
        max_lat = bounding_box.get("max_lat", ds.latitude.values.max())
        min_lon = bounding_box.get("min_lon", ds.longitude.values.min())
        max_lon = bounding_box.get("max_lon", ds.longitude.values.max())

        ds_sub = ds.sel(
            latitude=slice(min_lat, max_lat),
            longitude=slice(min_lon, max_lon),
        )
        if len(ds_sub.latitude) == 0 or len(ds_sub.longitude) == 0:
            ds_sub = ds

        return self.downscale_grid(ds_sub, variable_name, target_resolution_km=target_res)
