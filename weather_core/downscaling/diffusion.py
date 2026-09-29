import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from typing import Dict, Any, Tuple, Optional
from .diffusion_interface import ConditionalDiffusionDownscaler


class GaussianDiffusionScheduler:
    """
    DDPM Linear/Cosine Noise Scheduler for Forward Noising and Reverse Sampling.
    """

    def __init__(self, timesteps: int = 100, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: Gradually adds Gaussian noise to target field x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_start.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_start.device)

        x_noisy = sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
        return x_noisy, noise


class ResBlock2D(nn.Module):
    """Lightweight 2D Residual block for meteorological feature maps."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.shortcut(x))


class ConditionalWeatherUNet(nn.Module):
    """
    Lightweight 2D U-Net architecture for conditional meteorological diffusion downscaling.
    Accepts concatenated [x_noisy, coarse_condition].
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        # Encoder
        self.enc1 = ResBlock2D(in_channels, base_channels)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResBlock2D(base_channels * 2, base_channels * 2)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = ResBlock2D(base_channels * 4, base_channels * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock2D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock2D(base_channels * 2, base_channels)

        # Output head
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_noisy: torch.Tensor, coarse_condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_noisy: [B, 1, H, W]
        coarse_condition: [B, 1, H, W]
        t: [B]
        """
        # Concatenate noisy target and coarse conditioning input
        x = torch.cat([x_noisy, coarse_condition], dim=1)  # [B, 2, H, W]

        # Encoder path
        e1 = self.enc1(x)  # [B, C, H, W]
        d1 = self.down1(e1)  # [B, 2C, H/2, W/2]

        e2 = self.enc2(d1)  # [B, 2C, H/2, W/2]
        d2 = self.down2(e2)  # [B, 4C, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(d2)

        # Decoder path with skip connections
        u2 = self.up2(b)  # [B, 2C, H/2, W/2]
        # Match spatial shapes for skip connection if padding differs by 1
        if u2.shape != e2.shape:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2_out = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2_out)  # [B, C, H, W]
        if u1.shape != e1.shape:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1_out = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final_conv(d1_out)


class ConditionalWeatherDiffusion(ConditionalDiffusionDownscaler):
    """
    Complete DDPM Conditional Weather Diffusion Downscaler wrapper.
    """

    def __init__(
        self,
        target_resolution_km: float = 5.0,
        timesteps: int = 100,
        base_channels: int = 16,
        model: Optional[nn.Module] = None,
        scheduler: Optional[GaussianDiffusionScheduler] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(target_resolution_km=target_resolution_km, num_sampling_steps=timesteps)
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = scheduler if scheduler is not None else GaussianDiffusionScheduler(timesteps=timesteps)
        self.model = model if model is not None else ConditionalWeatherUNet(in_channels=2, base_channels=base_channels)
        self.model.to(self.device)

    def sample(self, coarse_cond: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Direct tensor reverse diffusion sampling given coarse conditioning tensor.
        """
        self.model.eval()
        steps = num_steps if num_steps is not None else self.num_sampling_steps
        B, C, H, W = coarse_cond.shape
        x_t = torch.randn((B, 1, H, W), device=self.device)

        for step in reversed(range(steps)):
            t_tensor = torch.tensor([step] * B, device=self.device, dtype=torch.long)
            pred_noise = self.model(x_t, coarse_cond, t_tensor)

            beta = self.scheduler.betas[step].to(self.device)
            alpha = self.scheduler.alphas[step].to(self.device)
            sqrt_one_minus_alpha_bar = self.scheduler.sqrt_one_minus_alphas_cumprod[step].to(self.device)

            x_t = (1.0 / torch.sqrt(alpha)) * (x_t - (beta / sqrt_one_minus_alpha_bar) * pred_noise)
            if step > 0:
                x_t = x_t + torch.sqrt(beta) * torch.randn_like(x_t)

        return x_t

    def sample_high_res_field(
        self,
        coarse_field: xr.DataArray,
        static_topography: Optional[np.ndarray] = None,
        event_embeddings: Optional[np.ndarray] = None,
        conditioning_vars: Optional[Dict[str, Any]] = None,
        num_ensembles: int = 1,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Performs reverse diffusion sampling conditioned on coarse weather field.
        """
        self.model.eval()

        coarse_vals = coarse_field.values
        if coarse_vals.ndim == 3:
            coarse_vals = coarse_vals[0]

        # Interpolate coarse field to target fine resolution shape for conditioning
        coarse_tensor = torch.tensor(coarse_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        # Resample fine shape 4x
        H, W = coarse_vals.shape[0] * 4, coarse_vals.shape[1] * 4
        coarse_cond = F.interpolate(coarse_tensor, size=(H, W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            x_t = self.sample(coarse_cond, num_steps=self.num_sampling_steps)

        samples_np = x_t.cpu().numpy()[:, 0, :, :]

        meta = {
            "method": "Conditional Weather DDPM Diffusion Sampling",
            "target_resolution_km": self.target_resolution_km,
            "sampling_steps": self.num_sampling_steps,
            "num_ensembles": num_ensembles,
            "compute_device": str(self.device),
            "disclaimer": self.DISCLAIMER,
        }

        return samples_np, meta


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Timestep Embedding for Diffusion Models.
    Maps discrete timestep integer t -> continuous high-dimensional vector.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeConditionedResBlock2D(nn.Module):
    """
    2D Residual Block with Sinusoidal Timestep Projection.
    Injects time embedding t_emb into spatial feature maps prior to second convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        if t_emb is not None:
            time_feat = self.time_proj(F.gelu(t_emb))
            h = h + time_feat.unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.shortcut(x))


class SR3WeatherUNet(nn.Module):
    """
    AERIS SR3-Inspired Conditional Diffusion U-Net.
    Features:
    - Sinusoidal timestep embedding injected into every encoder, bottleneck, and decoder ResBlock.
    - Low-resolution coarse conditioning concatenated with noisy target [x_t, y_coarse].
    - Deep residual feature extraction.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder with time conditioning
        self.enc1 = TimeConditionedResBlock2D(in_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.enc2 = TimeConditionedResBlock2D(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = TimeConditionedResBlock2D(base_channels * 4, base_channels * 4, time_dim)

        # Decoder with skip connections and time conditioning
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = TimeConditionedResBlock2D(base_channels * 4, base_channels * 2, time_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = TimeConditionedResBlock2D(base_channels * 2, base_channels, time_dim)

        # Output head
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_noisy: torch.Tensor, coarse_condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = torch.cat([x_noisy, coarse_condition], dim=1)

        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)

        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)

        b = self.bottleneck(d2, t_emb)

        u2 = self.up2(b)
        if u2.shape != e2.shape:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2_out = self.dec2(torch.cat([u2, e2], dim=1), t_emb)

        u1 = self.up1(d2_out)
        if u1.shape != e1.shape:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1_out = self.dec1(torch.cat([u1, e1], dim=1), t_emb)

        return self.final_conv(d1_out)


class AERISSR3ConditionalWeatherDiffusion(ConditionalWeatherDiffusion):
    """
    Enhanced AERIS SR3-Inspired Conditional Diffusion Downscaler.
    Combines Sinusoidal Timestep Conditioning with SR3 Residual Conditioning.
    """

    def __init__(
        self,
        target_resolution_km: float = 5.0,
        timesteps: int = 100,
        base_channels: int = 16,
        model: Optional[nn.Module] = None,
        scheduler: Optional[GaussianDiffusionScheduler] = None,
        device: Optional[torch.device] = None,
    ):
        device_obj = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sched_obj = scheduler if scheduler is not None else GaussianDiffusionScheduler(timesteps=timesteps)
        model_obj = model if model is not None else SR3WeatherUNet(in_channels=2, base_channels=base_channels)
        super().__init__(
            target_resolution_km=target_resolution_km,
            timesteps=timesteps,
            base_channels=base_channels,
            model=model_obj,
            scheduler=sched_obj,
            device=device_obj,
        )

