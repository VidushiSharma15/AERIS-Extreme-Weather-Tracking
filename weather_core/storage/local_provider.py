import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from .base import StorageProvider


class LocalStorageProvider(StorageProvider):
    """
    Concrete LocalStorageProvider operating on local filesystem.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, key: str) -> Path:
        path = Path(key)
        if path.is_absolute():
            return path
        return self.root_dir / path

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        src = Path(local_path)
        dest = self._resolve_path(remote_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return True

    def download_file(self, remote_key: str, local_destination: str) -> str:
        src = self._resolve_path(remote_key)
        dest = Path(local_destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return str(dest.resolve())

    def list_files(self, prefix: str = "") -> List[str]:
        target = self._resolve_path(prefix)
        if not target.exists():
            return []
        if target.is_file():
            return [str(target.relative_to(self.root_dir))]

        files = []
        for p in target.rglob("*"):
            if p.is_file():
                try:
                    rel = str(p.relative_to(self.root_dir))
                except ValueError:
                    rel = str(p)
                files.append(rel)
        return files

    def file_exists(self, key: str) -> bool:
        return self._resolve_path(key).is_file()

    def get_file_size_bytes(self, key: str) -> int:
        path = self._resolve_path(key)
        if not path.exists():
            return 0
        return path.stat().st_size

    def get_metadata(self, key: str) -> Dict[str, Any]:
        path = self._resolve_path(key)
        if not path.exists():
            return {}
        stat = path.stat()
        return {
            "key": key,
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "modified_time": stat.st_mtime,
            "storage_type": "local",
        }

    def compute_sha256(self, key: str) -> str:
        path = self._resolve_path(key)
        if not path.exists():
            return ""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
