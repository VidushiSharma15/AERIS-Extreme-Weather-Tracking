from pathlib import Path
import xarray as xr
from .base import WeatherDataProvider
from .validator import DatasetValidationError


class LocalNetCDFProvider(WeatherDataProvider):
    """
    LocalNetCDFProvider reads local .nc / NetCDF files using xarray netcdf4/h5netcdf engine.
    """

    def __init__(self):
        super().__init__(provider_name="LocalNetCDF")

    def open_dataset(self, source_path: str) -> xr.Dataset:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"NetCDF weather file not found at: {source_path}")

        try:
            return xr.open_dataset(path, engine="netcdf4")
        except Exception:
            try:
                return xr.open_dataset(path, engine="h5netcdf")
            except Exception as e:
                raise DatasetValidationError(f"Failed to open NetCDF file '{source_path}': {str(e)}")
