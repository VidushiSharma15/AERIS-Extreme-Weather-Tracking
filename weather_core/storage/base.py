from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, BinaryIO


class StorageProvider(ABC):
    """
    Abstract StorageProvider interface.
    Decouples local filesystem and cloud object storage.
    """

    @abstractmethod
    def upload_file(self, local_path: str, remote_key: str) -> bool:
        """Upload a file to the storage provider."""
        pass

    @abstractmethod
    def download_file(self, remote_key: str, local_destination: str) -> str:
        """Download a file from the storage provider to local destination."""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List file keys under a given prefix."""
        pass

    @abstractmethod
    def file_exists(self, key: str) -> bool:
        """Check if a file key exists."""
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Retrieve metadata for a file key (size, modification date, checksum)."""
        pass

    @abstractmethod
    def get_file_size_bytes(self, key: str) -> int:
        """Get file size in bytes."""
        pass
