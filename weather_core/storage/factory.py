from typing import Optional
from .base import StorageProvider
from .local_provider import LocalStorageProvider
from .cloud_provider import S3StorageProvider, GCSStorageProvider, AzureBlobStorageProvider
from ..config import get_settings


class StorageFactory:
    """
    Factory for retrieving configured StorageProvider instance.
    Decouples storage backend instantiation from business logic.
    """

    @staticmethod
    def get_provider(provider_type: Optional[str] = None, data_root: Optional[str] = None) -> StorageProvider:
        settings = get_settings()
        ptype = (provider_type or settings.provider_type).lower()

        if ptype == "local":
            root = data_root or settings.data_root
            return LocalStorageProvider(root_dir=root)
        elif ptype in ("s3", "aws"):
            return S3StorageProvider()
        elif ptype in ("gcs", "google"):
            return GCSStorageProvider()
        elif ptype in ("azure", "blob"):
            return AzureBlobStorageProvider()
        else:
            root = data_root or settings.data_root
            return LocalStorageProvider(root_dir=root)
