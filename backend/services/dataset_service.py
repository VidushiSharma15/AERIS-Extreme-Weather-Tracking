from pathlib import Path
from typing import Dict, Any, Optional
from weather_core.ingestion import LocalNetCDFProvider
from weather_core.data_manager import DatasetManager
from weather_core.storage import StorageFactory
from weather_core.config import get_settings


class DatasetService:
    """
    Service layer providing dataset status and metadata specifications.
    """

    def __init__(self):
        self.settings = get_settings()
        self.storage = StorageFactory.get_provider()
        self.dataset_manager = DatasetManager(storage_provider=self.storage)

    def get_data_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "environment": self.settings.environment,
            "provider_type": self.storage.__class__.__name__,
            "data_root": self.settings.data_root,
            "max_local_dataset_gb": self.settings.max_local_dataset_gb,
            "registered_datasets_count": len(self.dataset_manager.list_registered_datasets()),
            "registered_datasets": self.dataset_manager.list_registered_datasets(),
        }

    def get_metadata(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        grib2_path = Path("D:/SIH26078_AERIS/data/raw/nepsg/nepsg_amphan_2020.grib2")
        if grib2_path.exists():
            return {
                "dataset_name": "tigge-forecasts (dems NCMRWF)",
                "variable": "mean_sea_level_pressure",
                "units": "hPa",
                "source": "NCMRWF NEPS-G Forecast Archive",
                "geographical_bounds": {
                    "min_lat": 10.0,
                    "max_lat": 25.0,
                    "min_lon": 80.0,
                    "max_lon": 95.0,
                },
                "time_start": "2020-05-17 00:00 UTC",
                "time_end": "2020-05-20 00:00 UTC",
                "resolution_km": 55.0,
                "local_or_cloud": "local_storage",
                "file_location": str(grib2_path),
                "size_bytes": grib2_path.stat().st_size,
                "processing_version": "0.1.0-dev",
            }

        sample_path = Path(self.settings.sample_data_root) / "india_weather_sample.nc"
        if not sample_path.exists():
            sample_path = Path(self.settings.processed_data_root) / "india_weather_sample_standardized.nc"

        if not sample_path.exists():
            return {
                "dataset_name": dataset_name or "none",
                "status": "No active dataset loaded",
                "processing_version": "0.1.0-dev",
            }

        provider = LocalNetCDFProvider()
        meta = provider.extract_metadata(str(sample_path), dataset_name=dataset_name or sample_path.stem)
        meta_dict = meta.to_dict()
        meta_dict["processing_version"] = "0.1.0-dev"
        return meta_dict
