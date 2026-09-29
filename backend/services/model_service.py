from typing import Dict, Any
from weather_core.config import get_settings


class ModelService:
    """
    Service layer providing model status, device info, and downscaling metadata.
    """

    def __init__(self):
        self.settings = get_settings()

    def get_status(self) -> Dict[str, Any]:
        return {
            "device": self.settings.device,
            "downscaling_engine": "Baseline statistical/geospatial downscaling",
            "gnn_status": "GNN prototype interface ready",
            "diffusion_status": "ConditionalDiffusionDownscaler research schema ready",
        }
