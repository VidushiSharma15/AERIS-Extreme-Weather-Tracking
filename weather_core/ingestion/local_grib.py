from pathlib import Path
import xarray as xr
from .base import WeatherDataProvider
from .validator import DatasetValidationError


class LocalGRIBProvider(WeatherDataProvider):
    """
    LocalGRIBProvider reads local .grib / .grib2 weather files using cfgrib or NetCDF conversion.
    """

    def __init__(self):
        super().__init__(provider_name="LocalGRIB")

    def open_dataset(self, source_path: str) -> xr.Dataset:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"GRIB weather file not found at: {source_path}")

        try:
            return xr.open_dataset(path, engine="cfgrib")
        except Exception as e:
            # Fallback or informative exception if cfgrib engine is not installed
            raise DatasetValidationError(
                f"Failed to open GRIB file '{source_path}': {str(e)}. "
                f"Ensure GRIB files are converted to NetCDF or cfgrib engine is configured."
            )
