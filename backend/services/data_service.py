from typing import Dict, Any, List
from weather_core.data_manager import DatasetManager
from weather_core.storage import StorageFactory


class DataService:
    """
    Service layer providing data status and metadata access via storage abstraction.
    """

    def __init__(self):
        self.storage = StorageFactory.get_provider()
        self.dataset_manager = DatasetManager(storage_provider=self.storage)

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "online",
            "provider_type": self.storage.__class__.__name__,
            "datasets_registered": len(self.dataset_manager.list_registered_datasets()),
            "registered_datasets": self.dataset_manager.list_registered_datasets(),
        }
