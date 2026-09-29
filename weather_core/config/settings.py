import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml  # PyYAML if available
except ImportError:
    yaml = None


class Settings:
    """
    Central Settings class for AERIS.
    Parses YAML configuration files and environment overrides.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.config_path = config_path or os.environ.get(
            "CONFIG_PATH", str(self.root_dir / "configs" / "local.yaml")
        )
        self.config_data: Dict[str, Any] = self._load_config(self.config_path)

        # Environment & System Caches
        self.environment: str = os.environ.get("ENVIRONMENT", self.config_data.get("environment", "local"))
        self.debug: bool = str(os.environ.get("DEBUG", self.config_data.get("debug", True))).lower() in ("true", "1")

        # Storage paths
        storage_cfg = self.config_data.get("storage", {})
        if not isinstance(storage_cfg, dict):
            storage_cfg = {}

        self.provider_type: str = os.environ.get("STORAGE_PROVIDER_TYPE", storage_cfg.get("provider_type", "local"))
        self.data_root: str = os.environ.get("DATA_ROOT", storage_cfg.get("data_root", "D:/SIH26078_AERIS/data"))
        self.raw_data_root: str = os.environ.get("RAW_DATA_ROOT", storage_cfg.get("raw_data_root", f"{self.data_root}/raw"))
        self.processed_data_root: str = os.environ.get("PROCESSED_DATA_ROOT", storage_cfg.get("processed_data_root", f"{self.data_root}/processed"))
        self.sample_data_root: str = os.environ.get("SAMPLE_DATA_ROOT", storage_cfg.get("sample_data_root", f"{self.data_root}/samples"))
        self.metadata_root: str = os.environ.get("METADATA_ROOT", storage_cfg.get("metadata_root", f"{self.data_root}/metadata"))
        self.model_root: str = os.environ.get("MODEL_ROOT", storage_cfg.get("model_root", "D:/SIH26078_AERIS/models"))
        self.cache_root: str = os.environ.get("CACHE_ROOT", storage_cfg.get("cache_root", "D:/SIH26078_AERIS/cache"))
        self.log_root: str = os.environ.get("LOG_ROOT", storage_cfg.get("log_root", "D:/SIH26078_AERIS/logs"))

        # Storage Guardrails
        safety_cfg = self.config_data.get("safety", {})
        if not isinstance(safety_cfg, dict):
            safety_cfg = {}

        self.max_local_dataset_gb: float = float(
            os.environ.get("MAX_LOCAL_DATASET_GB", safety_cfg.get("max_local_dataset_gb", 2.0))
        )
        self.allow_large_downloads: bool = str(
            os.environ.get("ALLOW_LARGE_DOWNLOADS", safety_cfg.get("allow_large_downloads", False))
        ).lower() in ("true", "1")

        # Compute Device Selection
        compute_cfg = self.config_data.get("compute", {})
        if not isinstance(compute_cfg, dict):
            compute_cfg = {}
        requested_device = os.environ.get("DEVICE", compute_cfg.get("device", "auto"))
        self.device = self._resolve_device(requested_device)

    def _load_config(self, filepath: str) -> Dict[str, Any]:
        path = Path(filepath)
        if not path.is_absolute():
            path = self.root_dir / path

        if not path.exists():
            return {}

        if yaml is not None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass

        # Fallback simple block parser if PyYAML is not installed
        config: Dict[str, Any] = {}
        current_section: Optional[str] = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw_line = line
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                indent = len(raw_line) - len(raw_line.lstrip())
                if ":" in line:
                    k, v = line.split(":", 1)
                    key = k.strip()
                    val = v.strip()
                    # Strip inline comments if present outside quotes
                    if "#" in val and not (val.startswith('"') or val.startswith("'")):
                        val = val.split("#", 1)[0].strip()
                    val = val.strip('"').strip("'")

                    # Convert types
                    if val.lower() == "true":
                        typed_val = True
                    elif val.lower() == "false":
                        typed_val = False
                    else:
                        try:
                            typed_val = float(val) if "." in val else int(val)
                        except ValueError:
                            typed_val = val

                    if indent == 0:
                        if not val:
                            current_section = key
                            config[current_section] = {}
                        else:
                            current_section = None
                            config[key] = typed_val
                    elif indent > 0 and current_section is not None:
                        if isinstance(config.get(current_section), dict):
                            config[current_section][key] = typed_val

        return config

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        return requested

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "debug": self.debug,
            "provider_type": self.provider_type,
            "data_root": self.data_root,
            "raw_data_root": self.raw_data_root,
            "processed_data_root": self.processed_data_root,
            "model_root": self.model_root,
            "cache_root": self.cache_root,
            "max_local_dataset_gb": self.max_local_dataset_gb,
            "allow_large_downloads": self.allow_large_downloads,
            "device": self.device,
        }


_settings_instance: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    global _settings_instance
    if _settings_instance is None or config_path is not None:
        _settings_instance = Settings(config_path=config_path)
    return _settings_instance
