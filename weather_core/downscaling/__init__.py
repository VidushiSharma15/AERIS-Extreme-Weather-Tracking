"""
Downscaling module — Baseline geospatial interpolation & conditional diffusion research interface (0.5deg -> 5km).
"""

from .base import BaseDownscaler
from .metrics import AmplitudePreservationMetrics
from .baseline import BilinearDownscaler, StatisticalRefinementDownscaler
from .diffusion_interface import ConditionalDiffusionDownscaler, PhysicsConstraint
from .diffusion import (
    ConditionalWeatherDiffusion,
    GaussianDiffusionScheduler,
    ConditionalWeatherUNet,
    SinusoidalPositionalEmbedding,
    TimeConditionedResBlock2D,
    SR3WeatherUNet,
    AERISSR3ConditionalWeatherDiffusion,
)
from .physics_loss import (
    PhysicsInformedLoss,
    GradientAwareLoss,
    QuantileLoss,
    ExtremeAwareLoss,
    SpectralLoss,
    AERISCombinedLoss,
)
from .evaluator import DownscalingEvaluator
from .ablation import DownscalingAblationEvaluator

__all__ = [
    "BaseDownscaler",
    "AmplitudePreservationMetrics",
    "BilinearDownscaler",
    "StatisticalRefinementDownscaler",
    "ConditionalDiffusionDownscaler",
    "PhysicsConstraint",
    "ConditionalWeatherDiffusion",
    "GaussianDiffusionScheduler",
    "ConditionalWeatherUNet",
    "SinusoidalPositionalEmbedding",
    "TimeConditionedResBlock2D",
    "SR3WeatherUNet",
    "AERISSR3ConditionalWeatherDiffusion",
    "PhysicsInformedLoss",
    "GradientAwareLoss",
    "QuantileLoss",
    "ExtremeAwareLoss",
    "SpectralLoss",
    "AERISCombinedLoss",
    "DownscalingEvaluator",
    "DownscalingAblationEvaluator",
]
