from typing import List, Dict, Any
from .base import StorageProvider


class AbstractCloudStorageProvider(StorageProvider):
    """
    Abstract interface for Cloud Object Storage (S3 / GCS / Azure Blob).
    Allows local development without requiring external cloud SDK installations.
    """

    def __init__(self, bucket_name: str, region: str = "ap-south-1"):
        self.bucket_name = bucket_name
        self.region = region

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        raise NotImplementedError(
            f"Cloud upload to {self.bucket_name}/{remote_key} requires cloud SDK integration."
        )

    def download_file(self, remote_key: str, local_destination: str) -> str:
        raise NotImplementedError(
            f"Cloud download from {self.bucket_name}/{remote_key} requires cloud SDK integration."
        )

    def list_files(self, prefix: str = "") -> List[str]:
        return []

    def file_exists(self, key: str) -> bool:
        return False

    def get_file_size_bytes(self, key: str) -> int:
        return 0

    def get_metadata(self, key: str) -> Dict[str, Any]:
        return {
            "key": key,
            "bucket": self.bucket_name,
            "region": self.region,
            "storage_type": "cloud_abstract",
        }


class S3StorageProvider(AbstractCloudStorageProvider):
    """AWS S3 Storage Provider Interface."""

    def __init__(self, bucket_name: str = "sih26078-aeris-data", region: str = "ap-south-1"):
        super().__init__(bucket_name=bucket_name, region=region)


class GCSStorageProvider(AbstractCloudStorageProvider):
    """Google Cloud Storage Provider Interface."""

    def __init__(self, bucket_name: str = "sih26078-aeris-data", region: str = "asia-south1"):
        super().__init__(bucket_name=bucket_name, region=region)


class AzureBlobStorageProvider(AbstractCloudStorageProvider):
    """Azure Blob Storage Provider Interface."""

    def __init__(self, bucket_name: str = "sih26078-aeris-data", region: str = "centralindia"):
        super().__init__(bucket_name=bucket_name, region=region)
