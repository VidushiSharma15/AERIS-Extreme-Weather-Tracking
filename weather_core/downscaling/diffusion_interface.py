from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
from typing import Dict, Any, Optional, Tuple


class ConditionalDiffusionDownscaler(ABC):
    """
    Interface definition for future Conditional Generative Score-Based Diffusion Downscaling.

    DISCLAIMER:
    Interface definition only. Conditional diffusion model training and validated 5-km
    probabilistic forecasting remain a research extension.
    """

    DISCLAIMER = (
        "Conditional Diffusion Downscaler interface; model training and validated "
        "5-km probabilistic generative forecasting remain a research extension."
    )

    def __init__(self, target_resolution_km: float = 5.0, num_sampling_steps: int = 50):
        self.target_resolution_km = target_resolution_km
        self.num_sampling_steps = num_sampling_steps

    @abstractmethod
    def sample_high_res_field(
        self,
        coarse_field: xr.DataArray,
        static_topography: Optional[np.ndarray] = None,
        event_embeddings: Optional[np.ndarray] = None,
        conditioning_vars: Optional[Dict[str, Any]] = None,
        num_ensembles: int = 10,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Abstract sampler interface producing high-resolution probabilistic downscaled ensembles.
        Returns (ensemble_samples_array [num_ensembles, H, W], metadata_dict).
        """
        pass


class PhysicsConstraint(ABC):
    """
    Interface definition for physics-informed consistency constraints during downscaling.
    Supports mass, moisture, and thermodynamic consistency enforcement.
    """

    @abstractmethod
    def enforce_mass_conservation(
        self, coarse_field: np.ndarray, fine_field: np.ndarray
    ) -> np.ndarray:
        """Enforces spatial mass & moisture flux conservation across fine grid cells."""
        pass

    @abstractmethod
    def enforce_hydrostatic_balance(
        self, pressure_field: np.ndarray, temperature_field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Enforces atmospheric hydrostatic and thermodynamic consistency."""
        pass
