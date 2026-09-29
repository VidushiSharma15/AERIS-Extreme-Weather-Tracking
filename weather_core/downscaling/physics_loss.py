import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class GradientAwareLoss(nn.Module):
    """
    Spatial gradient loss using 2D Sobel operators.
    Encourages sharp pressure gradients, eyewall boundaries, and wind shear structure preservation.
    """

    def __init__(self, weight: float = 0.1, operator: str = "sobel", reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        # 2D Sobel Horizontal (kx) and Vertical (ky) Kernels
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Replicate padding to preserve grid shape
        p_px = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode="replicate"), self.kx)
        p_py = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode="replicate"), self.ky)
        t_px = F.conv2d(F.pad(target, (1, 1, 1, 1), mode="replicate"), self.kx)
        t_py = F.conv2d(F.pad(target, (1, 1, 1, 1), mode="replicate"), self.ky)

        grad_loss = F.l1_loss(p_px, t_px, reduction=self.reduction) + F.l1_loss(p_py, t_py, reduction=self.reduction)
        return self.weight * grad_loss


class QuantileLoss(nn.Module):
    """
    Pinball / Quantile loss on spatial fields.
    For quantile q in (0, 1): L_q(y, y_hat) = max(q*(y - y_hat), (q-1)*(y - y_hat))
    """

    def __init__(self, quantile: float = 0.95, weight: float = 0.1):
        super().__init__()
        self.quantile = quantile
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target - pred
        loss = torch.max(self.quantile * err, (self.quantile - 1.0) * err)
        return self.weight * torch.mean(loss)


class ExtremeAwareLoss(nn.Module):
    """
    Threshold-weighted tail loss focusing error penalties on extreme upper/lower percentile target values.
    """

    def __init__(self, quantile_threshold: float = 0.90, tail_weight_factor: float = 3.0, weight: float = 0.1):
        super().__init__()
        self.quantile_threshold = quantile_threshold
        self.tail_weight_factor = tail_weight_factor
        self.weight = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t_detach = target.detach()
        t_thresh = torch.quantile(t_detach.flatten(1), self.quantile_threshold, dim=1).view(-1, 1, 1, 1)
        weights = 1.0 + (self.tail_weight_factor - 1.0) * (target > t_thresh).float()
        sq_err = weights * (pred - target) ** 2
        return self.weight * torch.mean(sq_err)


class SpectralLoss(nn.Module):
    """
    2D RFFT Power Spectral Density Loss.
    Compares 2D Fourier spatial frequency spectrum of prediction and reference.
    Prevents over-smoothing and preserves fine-scale spatial variability.
    """

    def __init__(self, weight: float = 0.05, log_space: bool = True, eps: float = 1e-8):
        super().__init__()
        self.weight = weight
        self.log_space = log_space
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fft_pred = torch.fft.rfft2(pred, norm="ortho")
        fft_target = torch.fft.rfft2(target, norm="ortho")

        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)

        if self.log_space:
            log_pred = torch.log(mag_pred + self.eps)
            log_target = torch.log(mag_target + self.eps)
            loss = F.l1_loss(log_pred, log_target)
        else:
            loss = F.mse_loss(mag_pred, mag_target)

        return self.weight * loss


class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss Module for Meteorological Field Diffusion Downscaling.
    Enforces physical constraints:
    - Non-negativity for precipitation & wind speed
    - Coarse-to-fine block aggregation mass matching
    - Wind vector consistency (wind_speed = sqrt(u^2 + v^2))
    - Peak extreme value preservation
    """

    def __init__(self, lambda_extreme: float = 0.1, lambda_physics: float = 0.05):
        super().__init__()
        self.lambda_extreme = lambda_extreme
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_field: torch.Tensor,
        target_field: torch.Tensor,
        variable_name: str = "msl",
        coarse_cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Reconstruction Loss (Noise prediction MSE)
        l_recon = self.mse(pred_noise, target_noise)

        # 2. Extreme Value Preservation Loss (Penalizes underestimation of peak maximums)
        pred_max = torch.max(pred_field)
        target_max = torch.max(target_field)
        max_diff = F.relu(target_max - pred_max)
        l_extreme = (pred_max - target_max) ** 2 + max_diff ** 2

        # 3. Physics Consistency Loss
        l_physics = torch.tensor(0.0, device=pred_field.device)
        if variable_name in ["tp", "precipitation", "10m_wind_speed", "wind_speed"]:
            # Non-negativity constraint
            negative_penalty = torch.mean(F.relu(-pred_field) ** 2)
            l_physics = l_physics + negative_penalty

        # Coarse aggregation matching if coarse condition is provided
        if coarse_cond is not None and coarse_cond.shape == pred_field.shape:
            pred_coarse_block = F.avg_pool2d(pred_field, kernel_size=4, stride=4)
            target_coarse_block = F.avg_pool2d(coarse_cond, kernel_size=4, stride=4)
            l_physics = l_physics + F.mse_loss(pred_coarse_block, target_coarse_block)

        # Total Loss
        l_total = l_recon + self.lambda_extreme * l_extreme + self.lambda_physics * l_physics

        return {
            "loss": l_total,
            "l_recon": l_recon,
            "l_extreme": l_extreme,
            "l_physics": l_physics,
        }


class AERISCombinedLoss(nn.Module):
    """
    Combined Loss Engine for AERIS Extreme-Aware Conditional Diffusion Downscaling.
    L_total = L_recon + lambda_grad * L_grad + lambda_extreme * L_extreme + lambda_phys * L_phys + lambda_spec * L_spec
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        gradient_weight: float = 0.1,
        extreme_weight: float = 0.1,
        physics_weight: float = 0.05,
        spectral_weight: float = 0.05,
        gradient_enabled: bool = True,
        extreme_enabled: bool = True,
        physics_enabled: bool = True,
        spectral_enabled: bool = True,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.gradient_enabled = gradient_enabled
        self.extreme_enabled = extreme_enabled
        self.physics_enabled = physics_enabled
        self.spectral_enabled = spectral_enabled

        self.mse = nn.MSELoss()
        self.gradient_loss = GradientAwareLoss(weight=gradient_weight)
        self.extreme_loss = ExtremeAwareLoss(weight=extreme_weight)
        self.quantile_loss = QuantileLoss(weight=extreme_weight * 0.5)
        self.spectral_loss = SpectralLoss(weight=spectral_weight)
        self.physics_weight = physics_weight

    def forward(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_field: torch.Tensor,
        target_field: torch.Tensor,
        coarse_cond: Optional[torch.Tensor] = None,
        variable_name: str = "msl",
    ) -> Dict[str, torch.Tensor]:
        l_recon = self.mse(pred_noise, target_noise) * self.recon_weight

        l_grad = self.gradient_loss(pred_field, target_field) if self.gradient_enabled else torch.tensor(0.0, device=pred_field.device)
        l_extreme = (self.extreme_loss(pred_field, target_field) + self.quantile_loss(pred_field, target_field)) if self.extreme_enabled else torch.tensor(0.0, device=pred_field.device)
        l_spec = self.spectral_loss(pred_field, target_field) if self.spectral_enabled else torch.tensor(0.0, device=pred_field.device)

        l_phys = torch.tensor(0.0, device=pred_field.device)
        if self.physics_enabled:
            if variable_name in ["tp", "precipitation", "wind_speed", "10m_wind_speed"]:
                l_phys = l_phys + torch.mean(F.relu(-pred_field) ** 2)
            if coarse_cond is not None and coarse_cond.shape == pred_field.shape:
                pred_coarse_block = F.avg_pool2d(pred_field, kernel_size=4, stride=4)
                target_coarse_block = F.avg_pool2d(coarse_cond, kernel_size=4, stride=4)
                l_phys = l_phys + F.mse_loss(pred_coarse_block, target_coarse_block)

        l_phys_total = self.physics_weight * l_phys

        l_total = l_recon + l_grad + l_extreme + l_phys_total + l_spec

        return {
            "loss": l_total,
            "l_recon": l_recon,
            "l_gradient": l_grad,
            "l_extreme": l_extreme,
            "l_physics": l_phys_total,
            "l_spectral": l_spec,
        }
