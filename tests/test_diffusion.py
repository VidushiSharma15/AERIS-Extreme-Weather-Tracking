import unittest
import torch
from fastapi.testclient import TestClient

from backend.main import app
from weather_core.downscaling import (
    ConditionalWeatherUNet,
    GaussianDiffusionScheduler,
    ConditionalWeatherDiffusion,
    PhysicsInformedLoss,
    SR3WeatherUNet,
    AERISSR3ConditionalWeatherDiffusion,
    GradientAwareLoss,
    QuantileLoss,
    ExtremeAwareLoss,
    SpectralLoss,
    AERISCombinedLoss,
    DownscalingAblationEvaluator,
)


class TestDiffusionDownscaling(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = ConditionalWeatherUNet(in_channels=2, base_channels=8).to(self.device)
        self.scheduler = GaussianDiffusionScheduler(timesteps=50)
        self.diffusion = ConditionalWeatherDiffusion(model=self.model, scheduler=self.scheduler, device=self.device)
        self.loss_fn = PhysicsInformedLoss(lambda_extreme=0.1, lambda_physics=0.05)
        self.client = TestClient(app)

    def test_unet_forward_shape(self):
        x_noisy = torch.randn(1, 1, 32, 32)
        coarse_cond = torch.randn(1, 1, 32, 32)
        t = torch.tensor([10], dtype=torch.long)

        out = self.model(x_noisy, coarse_cond, t)
        self.assertEqual(out.shape, (1, 1, 32, 32))

    def test_diffusion_q_sample(self):
        x_start = torch.randn(1, 1, 32, 32)
        t = torch.tensor([5], dtype=torch.long)
        noise = torch.randn_like(x_start)

        x_noisy, target_noise = self.scheduler.q_sample(x_start, t, noise)
        self.assertEqual(x_noisy.shape, (1, 1, 32, 32))
        self.assertEqual(target_noise.shape, (1, 1, 32, 32))

    def test_physics_loss(self):
        pred_noise = torch.randn(1, 1, 16, 16)
        target_noise = torch.randn(1, 1, 16, 16)
        x_noisy = torch.randn(1, 1, 16, 16)
        x_start = torch.randn(1, 1, 16, 16)

        loss_dict = self.loss_fn(pred_noise, target_noise, x_noisy, x_start, variable_name="msl")
        self.assertIn("loss", loss_dict)
        self.assertIn("l_recon", loss_dict)
        self.assertIn("l_extreme", loss_dict)
        self.assertIn("l_physics", loss_dict)
        self.assertGreater(loss_dict["loss"].item(), 0.0)

    def test_diffusion_sampling(self):
        coarse_cond = torch.randn(1, 1, 16, 16)
        out = self.diffusion.sample(coarse_cond, num_steps=5)
        self.assertEqual(out.shape, (1, 1, 16, 16))

    def test_api_downscale_diffusion(self):
        response = self.client.post(
            "/api/v1/downscale/diffusion",
            json={
                "variable_name": "msl",
                "timesteps": 5,
                "bounding_box": {"min_lat": 12.0, "max_lat": 24.0, "min_lon": 82.0, "max_lon": 92.0},
            },
        )
        self.assertEqual(response.status_code, 200)
        res_data = response.json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["method"], "Conditional Weather UNet DDPM Diffusion")
        self.assertIn("amplitude_metrics", res_data)


class TestEnhancedDiffusionTechnologies(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.sr3_model = SR3WeatherUNet(in_channels=2, base_channels=8).to(self.device)
        self.sr3_diffusion = AERISSR3ConditionalWeatherDiffusion(model=self.sr3_model, timesteps=10, device=self.device)

    def test_sr3_timestep_embedding_changes_features(self):
        """Verify that sinusoidal timestep conditioning actively produces different network outputs for t1 != t2."""
        x_noisy = torch.randn(1, 1, 16, 16)
        coarse_cond = torch.randn(1, 1, 16, 16)
        t1 = torch.tensor([5], dtype=torch.long)
        t2 = torch.tensor([45], dtype=torch.long)

        out1 = self.sr3_model(x_noisy, coarse_cond, t1)
        out2 = self.sr3_model(x_noisy, coarse_cond, t2)

        self.assertEqual(out1.shape, (1, 1, 16, 16))
        self.assertEqual(out2.shape, (1, 1, 16, 16))
        diff = torch.abs(out1 - out2).max().item()
        self.assertGreater(diff, 1e-4, "Timestep conditioning failed to modify network features.")

    def test_gradient_aware_loss(self):
        """Test GradientAwareLoss 2D Sobel computation and gradient flow."""
        grad_loss_fn = GradientAwareLoss(weight=0.1)
        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.randn(1, 1, 16, 16)

        loss = grad_loss_fn(pred, target)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(torch.abs(pred.grad).sum().item(), 0.0)

    def test_quantile_and_extreme_aware_loss(self):
        """Test QuantileLoss and ExtremeAwareLoss spatial tail penalization."""
        quantile_loss_fn = QuantileLoss(quantile=0.95, weight=0.1)
        extreme_loss_fn = ExtremeAwareLoss(quantile_threshold=0.90, weight=0.1)

        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.randn(1, 1, 16, 16)

        q_loss = quantile_loss_fn(pred, target)
        ext_loss = extreme_loss_fn(pred, target)

        self.assertGreater(q_loss.item(), 0.0)
        self.assertGreater(ext_loss.item(), 0.0)

        total_l = q_loss + ext_loss
        total_l.backward()
        self.assertIsNotNone(pred.grad)

    def test_spectral_loss(self):
        """Test SpectralLoss 2D RFFT frequency-domain loss."""
        spectral_fn = SpectralLoss(weight=0.05)
        pred = torch.randn(1, 1, 16, 16, requires_grad=True)
        target = torch.randn(1, 1, 16, 16)

        loss = spectral_fn(pred, target)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_aeris_combined_loss(self):
        """Test AERISCombinedLoss multi-objective integration and component toggling."""
        combined_fn = AERISCombinedLoss(
            recon_weight=1.0,
            gradient_weight=0.1,
            extreme_weight=0.1,
            physics_weight=0.05,
            spectral_weight=0.05,
            gradient_enabled=True,
            extreme_enabled=True,
            physics_enabled=True,
            spectral_enabled=True,
        )

        pred_noise = torch.randn(1, 1, 16, 16)
        target_noise = torch.randn(1, 1, 16, 16)
        pred_field = torch.randn(1, 1, 16, 16, requires_grad=True)
        target_field = torch.randn(1, 1, 16, 16)
        coarse_cond = torch.randn(1, 1, 16, 16)

        res = combined_fn(pred_noise, target_noise, pred_field, target_field, coarse_cond=coarse_cond, variable_name="tp")
        self.assertIn("loss", res)
        self.assertIn("l_recon", res)
        self.assertIn("l_gradient", res)
        self.assertIn("l_extreme", res)
        self.assertIn("l_physics", res)
        self.assertIn("l_spectral", res)
        self.assertGreater(res["loss"].item(), 0.0)

    def test_backward_gradient_flow_through_sr3_unet(self):
        """Verify end-to-end backpropagation through SR3 UNet parameters."""
        x_noisy = torch.randn(1, 1, 16, 16)
        coarse_cond = torch.randn(1, 1, 16, 16)
        t = torch.tensor([10], dtype=torch.long)

        pred_noise = self.sr3_model(x_noisy, coarse_cond, t)
        loss = torch.mean(pred_noise ** 2)
        loss.backward()

        param = next(self.sr3_model.parameters())
        self.assertIsNotNone(param.grad)
        self.assertGreater(torch.abs(param.grad).sum().item(), 0.0)

    def test_downscaling_ablation_evaluator(self):
        """Test DownscalingAblationEvaluator variant contribution logging."""
        pred_noise = torch.randn(1, 1, 16, 16)
        target_noise = torch.randn(1, 1, 16, 16)
        pred_field = torch.randn(1, 1, 16, 16)
        target_field = torch.randn(1, 1, 16, 16)

        ablation_res = DownscalingAblationEvaluator.evaluate_loss_variants(
            pred_noise, target_noise, pred_field, target_field, variable_name="msl"
        )

        self.assertIn("baseline_recon_loss", ablation_res)
        self.assertIn("variants", ablation_res)
        self.assertIn("variant_A_gradient", ablation_res["variants"])
        self.assertIn("variant_E_combined_all", ablation_res["variants"])

    def test_baseline_vs_enhanced_selectable(self):
        """Verify baseline ConditionalWeatherDiffusion and enhanced AERISSR3ConditionalWeatherDiffusion are both selectable."""
        baseline_model = ConditionalWeatherDiffusion(timesteps=5, device=self.device)
        enhanced_model = AERISSR3ConditionalWeatherDiffusion(timesteps=5, device=self.device)

        coarse_cond = torch.randn(1, 1, 16, 16)
        out_base = baseline_model.sample(coarse_cond, num_steps=3)
        out_enh = enhanced_model.sample(coarse_cond, num_steps=3)

        self.assertEqual(out_base.shape, (1, 1, 16, 16))
        self.assertEqual(out_enh.shape, (1, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
