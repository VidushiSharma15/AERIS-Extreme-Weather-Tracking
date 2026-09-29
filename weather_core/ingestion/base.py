from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import xarray as xr
from weather_core.data_manager import DatasetMetadata, GeographicalBounds
from weather_core.storage import StorageFactory
from .validator import DatasetValidator, DatasetValidationError


class WeatherDataProvider(ABC):
    """
    Abstract WeatherDataProvider base class.
    Decouples raw file formats (NetCDF, GRIB2, HDF5, Cloud Store) from scientific core.
    """

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.storage = StorageFactory.get_provider()

    @abstractmethod
    def open_dataset(self, source_path: str) -> xr.Dataset:
        """Opens raw weather data file into an xarray Dataset."""
        pass

    def validate(self, source_path: str) -> Dict[str, Any]:
        """Validates weather dataset against coordinate and unit standards."""
        ds = self.open_dataset(source_path)
        return DatasetValidator.validate_dataset(ds)

    def extract_metadata(self, source_path: str, dataset_name: str = "") -> DatasetMetadata:
        """Extracts dataset metadata schema from the opened dataset."""
        ds = self.open_dataset(source_path)
        val_info = self.validate(source_path)

        file_size = self.storage.get_file_size_bytes(source_path)
        first_var = list(ds.data_vars.keys())[0]
        units = str(ds[first_var].attrs.get("units", "unknown"))

        name = dataset_name or f"{self.provider_name}_{first_var}"
        bounds = GeographicalBounds(
            min_lat=val_info["lat_range"][0],
            max_lat=val_info["lat_range"][1],
            min_lon=val_info["lon_range"][0],
            max_lon=val_info["lon_range"][1],
        )

        return DatasetMetadata(
            dataset_name=name,
            variable=str(first_var),
            units=units,
            source=self.provider_name,
            geographical_bounds=bounds,
            time_start=val_info["time_range"][0],
            time_end=val_info["time_range"][1],
            resolution_km=val_info["resolution_km"],
            local_or_cloud="local",
            file_location=source_path,
            size_bytes=file_size,
        )

    def extract_field(
        self,
        source_path: str,
        variable_name: str,
        time_index: Optional[int] = None,
        spatial_bounds: Optional[Dict[str, float]] = None,
    ) -> xr.DataArray:
        """Extracts a specific variable field with optional spatial/temporal cropping."""
        ds = self.open_dataset(source_path)
        if variable_name not in ds:
            raise ValueError(f"Variable '{variable_name}' not found in dataset. Available: {list(ds.data_vars.keys())}")

        field = ds[variable_name]

        if time_index is not None and "time" in field.dims:
            field = field.isel(time=time_index)

        if spatial_bounds:
            lat_name = DatasetValidator.find_coordinate(ds, DatasetValidator.ALLOWED_LAT_NAMES)
            lon_name = DatasetValidator.find_coordinate(ds, DatasetValidator.ALLOWED_LON_NAMES)
            if lat_name and lon_name:
                field = field.sel(
                    {
                        lat_name: slice(spatial_bounds["min_lat"], spatial_bounds["max_lat"]),
                        lon_name: slice(spatial_bounds["min_lon"], spatial_bounds["max_lon"]),
                    }
                )
        return field
