import unittest
import os
import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from weather_core.downscaling.nepsg_phase13_pipeline import run_phase13_pipeline
from weather_core.downscaling.metrics import AmplitudePreservationMetrics
from weather_core.downscaling.physics_loss import PhysicsInformedLoss
from backend.main import app
import torch


class TestNEPSGPhase13(unittest.TestCase):
    """
    Test suite for AERIS Phase 13 Conditional Diffusion / Extreme-Weather Downscaling Pipeline.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.summary_path = Path("D:/SIH26078_AERIS/data/processed/nepsg_amphan_downscaling_summary.json")
        cls.nc_path = Path("D:/SIH26078_AERIS/data/processed/nepsg_amphan_downscaled_diffusion.nc")

    def test_01_phase13_pipeline_execution(self):
        """Verify end-to-end execution of Phase 13 conditional diffusion downscaling pipeline."""
        summary = run_phase13_pipeline()

        self.assertIsNotNone(summary)
        self.assertEqual(summary["mode"], "trained research prototype")
        self.assertEqual(summary["native_forecast_input"]["native_resolution"], "0.5° (~55 km)")
        self.assertEqual(summary["target_output_grid"]["grid_name"], "5 km prototype output grid")
        self.assertTrue(self.summary_path.exists())
        self.assertTrue(self.nc_path.exists())

    def test_02_api_downscaling_status(self):
        """Test GET /api/v1/downscaling/status endpoint."""
        response = self.client.get("/api/v1/downscaling/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "COMPLETE")
        self.assertIn("scientific_limitations", data)
        self.assertIn("5 km prototype output grid", data["target_output_grid"])

    def test_03_api_downscaling_amphan_metadata(self):
        """Test GET /api/v1/downscaling/nepsg/amphan endpoint."""
        response = self.client.get("/api/v1/downscaling/nepsg/amphan")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("extreme_amplitude_comparison", data)
        self.assertIn("diffusion_prototype_0_1deg", data["extreme_amplitude_comparison"])

    def test_04_api_post_downscale_nepsg(self):
        """Test POST /api/v1/downscale/nepsg endpoint."""
        response = self.client.post("/api/v1/downscale/nepsg", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "trained research prototype")

    def test_05_physics_loss_non_negativity(self):
        """Test PhysicsInformedLoss non-negativity penalty for physical variables."""
        loss_fn = PhysicsInformedLoss(lambda_extreme=0.1, lambda_physics=0.05)
        pred_noise = torch.randn(1, 1, 16, 16)
        target_noise = torch.randn(1, 1, 16, 16)
        pred_field = torch.tensor([[-5.0, 10.0], [3.0, -2.0]])  # Contains negative values
        target_field = torch.tensor([[0.0, 10.0], [3.0, 0.0]])

        loss_dict = loss_fn(pred_noise, target_noise, pred_field, target_field, variable_name="tp")
        self.assertGreater(float(loss_dict["l_physics"].item()), 0.0)

    def test_06_amplitude_preservation_metrics(self):
        """Test AmplitudePreservationMetrics calculation."""
        import numpy as np
        coarse = np.array([[990.0, 1000.0], [980.0, 995.0]])
        fine = np.array([[988.0, 1000.0], [978.0, 995.0]])
        metrics = AmplitudePreservationMetrics.calculate(coarse, fine)

        self.assertEqual(metrics.coarse_max, 1000.0)
        self.assertEqual(metrics.fine_max, 1000.0)
        self.assertGreater(metrics.peak_preservation_score, 0.9)


if __name__ == "__main__":
    unittest.main()
