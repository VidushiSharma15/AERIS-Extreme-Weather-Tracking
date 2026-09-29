import torch
import torch.nn as nn
from typing import Dict, Any
from .physics_loss import (
    GradientAwareLoss,
    ExtremeAwareLoss,
    QuantileLoss,
    SpectralLoss,
    AERISCombinedLoss,
)


class DownscalingAblationEvaluator:
    """
    Lightweight Evaluation Framework comparing baseline vs loss variant contributions:
    - Baseline: Standard L_diffusion (L_recon MSE)
    - Variant A: L_diffusion + L_gradient
    - Variant B: L_diffusion + L_extreme
    - Variant C: L_diffusion + L_physics
    - Variant D: L_diffusion + L_spectral
    - Variant E: Combined AERIS Loss (All Enabled)
    """

    @classmethod
    def evaluate_loss_variants(
        cls,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        pred_field: torch.Tensor,
        target_field: torch.Tensor,
        coarse_cond: torch.Tensor = None,
        variable_name: str = "msl",
    ) -> Dict[str, Any]:
        device = pred_field.device

        # Baseline
        mse_fn = nn.MSELoss()
        l_recon = mse_fn(pred_noise, target_noise)
        baseline_loss = float(l_recon.item())

        # Individual components
        grad_fn = GradientAwareLoss(weight=0.1).to(device)
        extreme_fn = ExtremeAwareLoss(weight=0.1).to(device)
        quantile_fn = QuantileLoss(weight=0.05).to(device)
        spectral_fn = SpectralLoss(weight=0.05).to(device)

        l_grad_val = float(grad_fn(pred_field, target_field).item())
        l_ext_val = float((extreme_fn(pred_field, target_field) + quantile_fn(pred_field, target_field)).item())
        l_spec_val = float(spectral_fn(pred_field, target_field).item())

        l_phys_val = 0.0
        if variable_name in ["tp", "precipitation", "wind_speed", "10m_wind_speed"]:
            l_phys_val += float(torch.mean(torch.relu(-pred_field) ** 2).item()) * 0.05

        combined_loss_fn = AERISCombinedLoss().to(device)
        combined_dict = combined_loss_fn(
            pred_noise, target_noise, pred_field, target_field, coarse_cond=coarse_cond, variable_name=variable_name
        )

        return {
            "baseline_recon_loss": baseline_loss,
            "variants": {
                "baseline": {"total_loss": baseline_loss, "components": {"l_recon": baseline_loss}},
                "variant_A_gradient": {"total_loss": baseline_loss + l_grad_val, "components": {"l_recon": baseline_loss, "l_gradient": l_grad_val}},
                "variant_B_extreme": {"total_loss": baseline_loss + l_ext_val, "components": {"l_recon": baseline_loss, "l_extreme": l_ext_val}},
                "variant_C_physics": {"total_loss": baseline_loss + l_phys_val, "components": {"l_recon": baseline_loss, "l_physics": l_phys_val}},
                "variant_D_spectral": {"total_loss": baseline_loss + l_spec_val, "components": {"l_recon": baseline_loss, "l_spectral": l_spec_val}},
                "variant_E_combined_all": {
                    "total_loss": float(combined_dict["loss"].item()),
                    "components": {k: float(v.item()) for k, v in combined_dict.items() if k != "loss"},
                },
            },
        }
