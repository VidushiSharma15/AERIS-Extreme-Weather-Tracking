from typing import Dict, Any, Optional
import xarray as xr
from .base import WeatherDataProvider
from .local_netcdf import LocalNetCDFProvider
from weather_core.data_manager import DataSizeExceededError
from weather_core.config import get_settings


class BaseNWMProvider(WeatherDataProvider):
    """Base class for Numerical Weather Prediction (NWP) providers."""

    def __init__(self, provider_name: str, resolution_km: float):
        super().__init__(provider_name=provider_name)
        self.resolution_km = resolution_km
        self.settings = get_settings()
        self.local_delegate = LocalNetCDFProvider()

    def open_dataset(self, source_path: str) -> xr.Dataset:
        return self.local_delegate.open_dataset(source_path)

    def request_subset(
        self,
        variable: str,
        time_start: str,
        time_end: str,
        spatial_bounds: Dict[str, float],
        estimated_size_gb: float = 0.1,
    ) -> str:
        """
        Guards against multi-gigabyte dataset downloads.
        Returns path to regional NetCDF subset if local size is within limits.
        """
        if estimated_size_gb > self.settings.max_local_dataset_gb and not self.settings.allow_large_downloads:
            raise DataSizeExceededError(
                f"Requested {self.provider_name} subset ({estimated_size_gb:.2f} GB) exceeds "
                f"MAX_LOCAL_DATASET_GB limit ({self.settings.max_local_dataset_gb:.2f} GB). "
                f"Please narrow geographical bounds or request remote processing."
            )
        return f"{self.settings.raw_data_root}/{self.provider_name.lower()}_{variable}_subset.nc"


class ERA5Provider(BaseNWMProvider):
    """ERA5 Reanalysis Provider Interface."""
    def __init__(self):
        super().__init__(provider_name="ERA5", resolution_km=25.0)


class IMDAAProvider(BaseNWMProvider):
    """IMDAA Indian Regional Reanalysis Provider Interface (12 km)."""
    def __init__(self):
        super().__init__(provider_name="IMDAA", resolution_km=12.0)


class NCUMProvider(BaseNWMProvider):
    """NCMRWF Unified Model Global/Regional Forecast Provider Interface (12 km)."""
    def __init__(self):
        super().__init__(provider_name="NCUM", resolution_km=12.0)


class NEPSProvider(BaseNWMProvider):
    """NCMRWF Ensemble Prediction System Provider Interface (12 km, 23 ensemble members)."""
    def __init__(self, ensemble_member: Optional[int] = None):
        super().__init__(provider_name="NEPS", resolution_km=12.0)
        self.ensemble_member = ensemble_member
        self.data_source_type = "NWP_FORECAST"


class NEPSGProvider(BaseNWMProvider):
    """NCMRWF Global Ensemble Prediction System (NEPS-G) Provider Interface via TIGGE (origin=dems)."""
    def __init__(self, ensemble_members: Optional[list] = None):
        super().__init__(provider_name="NEPS-G", resolution_km=55.5)
        self.provider_wmo_code = "dems"
        self.ensemble_members = ensemble_members or list(range(12))
        self.data_source_type = "NWP_FORECAST"

    def open_dataset(self, source_path: str) -> xr.Dataset:
        if source_path.endswith((".grib2", ".grib")):
            from .grib2_decoder import decode_nepsg_grib2
            return decode_nepsg_grib2(source_path)
        return super().open_dataset(source_path)

