import unittest
import numpy as np
import pandas as pd
import xarray as xr
from weather_core.downscaling import (
    BilinearDownscaler,
    StatisticalRefinementDownscaler,
    AmplitudePreservationMetrics,
    DownscalingEvaluator,
    ConditionalDiffusionDownscaler,
    PhysicsConstraint,
)


class DummyDiffusionStub(ConditionalDiffusionDownscaler):
    def sample_high_res_field(self, coarse_field, static_topography=None, event_embeddings=None, conditioning_vars=None, num_ensembles=10):
        H, W = coarse_field.shape[-2] * 4, coarse_field.shape[-1] * 4
        return np.zeros((num_ensembles, H, W)), {"disclaimer": self.DISCLAIMER}


class DummyPhysicsStub(PhysicsConstraint):
    def enforce_mass_conservation(self, coarse_field, fine_field):
        return fine_field

    def enforce_hydrostatic_balance(self, pressure_field, temperature_field):
        return pressure_field, temperature_field


class TestDownscalingModule(unittest.TestCase):
    def setUp(self):
        lats = np.array([10.0, 15.0, 20.0, 25.0])
        lons = np.array([80.0, 85.0, 90.0, 95.0, 100.0])
        times = pd.date_range("2026-09-25T00:00:00", periods=2, freq="6h")

        t2m = np.random.uniform(295, 305, size=(2, 4, 5)).astype(np.float32)
        msl = np.random.uniform(94000, 101200, size=(2, 4, 5)).astype(np.float32)

        self.ds = xr.Dataset(
            data_vars={
                "t2m": (["time", "latitude", "longitude"], t2m),
                "msl": (["time", "latitude", "longitude"], msl),
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
        )

    def test_amplitude_preservation_metrics(self):
        c = np.array([[100.0, 150.0], [200.0, 250.0]])
        f = np.array([[100.0, 150.0], [200.0, 250.0]])
        metrics = AmplitudePreservationMetrics.calculate(c, f)

        self.assertEqual(metrics.coarse_max, 250.0)
        self.assertEqual(metrics.fine_max, 250.0)
        self.assertAlmostEqual(metrics.peak_ratio, 1.0, places=4)
        self.assertAlmostEqual(metrics.peak_preservation_score, 1.0, places=4)

    def test_bilinear_downscaler(self):
        downscaler = BilinearDownscaler(target_resolution_km=5.0)
        da_fine, meta = downscaler.downscale_grid(self.ds, "msl")

        self.assertEqual(meta["target_resolution_km"], 5.0)
        self.assertTrue(da_fine.shape[-2] > self.ds["msl"].shape[-2])
        self.assertTrue(da_fine.shape[-1] > self.ds["msl"].shape[-1])
        self.assertIn("amplitude_metrics", meta)

    def test_statistical_refinement_downscaler(self):
        downscaler = StatisticalRefinementDownscaler(target_resolution_km=5.0)
        da_fine, meta = downscaler.downscale_grid(self.ds, "msl")

        self.assertEqual(meta["target_resolution_km"], 5.0)
        self.assertIn("amplitude_metrics", meta)

    def test_downscaling_evaluator(self):
        c = np.array([10.0, 20.0, 30.0])
        f = np.array([10.0, 20.0, 30.0])
        res = DownscalingEvaluator.evaluate_downscaling_quality(c, f)

        self.assertAlmostEqual(res["spatial_mae"], 0.0, places=4)
        self.assertAlmostEqual(res["pearson_correlation"], 1.0, places=4)

    def test_diffusion_and_physics_interface_stubs(self):
        diff_stub = DummyDiffusionStub()
        phys_stub = DummyPhysicsStub()

        coarse_da = self.ds["msl"].isel(time=0)
        samples, meta = diff_stub.sample_high_res_field(coarse_da.values)
        self.assertEqual(samples.shape[0], 10)
        self.assertIn("disclaimer", meta)

        fine_mass = phys_stub.enforce_mass_conservation(coarse_da.values, coarse_da.values)
        self.assertEqual(fine_mass.shape, coarse_da.shape)


if __name__ == "__main__":
    unittest.main()
