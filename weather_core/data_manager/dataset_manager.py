import json
from pathlib import Path
from typing import Dict, List, Optional
from .metadata import DatasetMetadata
from ..storage.base import StorageProvider
from ..storage.factory import StorageFactory
from ..config import get_settings


class DataSizeExceededError(Exception):
    """Raised when a requested local dataset download/ingestion exceeds max_local_dataset_gb limit."""
    pass


class DatasetManager:
    """
    Manages dataset cataloging, metadata registration, and size safety guardrails.
    """

    def __init__(self, storage_provider: Optional[StorageProvider] = None):
        self.settings = get_settings()
        self.storage = storage_provider or StorageFactory.get_provider()
        self.metadata_catalog: Dict[str, DatasetMetadata] = {}

    def check_size_safety(self, size_bytes: int) -> bool:
        """
        Validates size against max_local_dataset_gb.
        Raises DataSizeExceededError if limit exceeded and allow_large_downloads is False.
        """
        size_gb = size_bytes / (1024 ** 3)
        max_allowed_gb = self.settings.max_local_dataset_gb

        if size_gb > max_allowed_gb and not self.settings.allow_large_downloads:
            raise DataSizeExceededError(
                f"Dataset size ({size_gb:.2f} GB) exceeds max allowed local limit ({max_allowed_gb:.2f} GB). "
                f"Please process remotely on the cloud or set ALLOW_LARGE_DOWNLOADS=true explicitly."
            )
        return True

    def register_dataset(self, metadata: DatasetMetadata) -> bool:
        """
        Registers dataset into catalog after validating size safety.
        """
        if metadata.local_or_cloud == "local":
            self.check_size_safety(metadata.size_bytes)

        self.metadata_catalog[metadata.dataset_name] = metadata
        self._save_metadata_manifest(metadata)
        return True

    def _save_metadata_manifest(self, metadata: DatasetMetadata):
        meta_dir = Path(self.settings.metadata_root)
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / f"{metadata.dataset_name}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def get_dataset(self, dataset_name: str) -> Optional[DatasetMetadata]:
        if dataset_name in self.metadata_catalog:
            return self.metadata_catalog[dataset_name]

        meta_file = Path(self.settings.metadata_root) / f"{dataset_name}_metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                meta = DatasetMetadata.from_dict(data)
                self.metadata_catalog[dataset_name] = meta
                return meta
        return None

    def list_registered_datasets(self) -> List[str]:
        meta_dir = Path(self.settings.metadata_root)
        if not meta_dir.exists():
            return list(self.metadata_catalog.keys())

        registered = set(self.metadata_catalog.keys())
        for p in meta_dir.glob("*_metadata.json"):
            name = p.stem.replace("_metadata", "")
            registered.add(name)
        return sorted(list(registered))
