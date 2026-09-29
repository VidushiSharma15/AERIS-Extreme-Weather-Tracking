from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
from typing import Dict, Any, Tuple, Optional


class BaseDownscaler(ABC):
    """
    Abstract base class defining the spatial downscaling contract.
    Refines coarse weather fields to localized target high-resolution representations.
    """

    DISCLAIMER = (
        "Baseline produces a 5-km representation through spatial refinement; "
        "it does not by itself create validated 5-km predictive information."
    )

    def __init__(self, target_resolution_km: float = 5.0):
        self.target_resolution_km = target_resolution_km

    @abstractmethod
    def downscale_grid(
        self,
        ds: xr.Dataset,
        variable_name: str,
        target_resolution_km: Optional[float] = None,
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        """
        Downscales a full spatial grid variable to target resolution.
        Returns (fine_data_array, metadata_dict).
        """
        pass

    @abstractmethod
    def downscale_event_region(
        self,
        ds: xr.Dataset,
        variable_name: str,
        bounding_box: Dict[str, float],
        target_resolution_km: Optional[float] = None,
    ) -> Tuple[xr.DataArray, Dict[str, Any]]:
        """
        Downscales a localized extreme event bounding box region to target resolution.
        Returns (fine_data_array, metadata_dict).
        """
        pass
