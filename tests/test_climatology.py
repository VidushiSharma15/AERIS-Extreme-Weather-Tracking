import unittest
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr

from weather_core.climatology.engine import ClimatologyEngine
from scripts.create_sample_weather_data import create_sample_weather_dataset


class TestClimatology(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sample_nc = Path(self.tmp_dir.name) / "test_sample.nc"
        create_sample_weather_dataset(output_path=str(self.sample_nc))
        self.ds = xr.open_dataset(self.sample_nc)

    def tearDown(self):
        self.ds.close()
        self.tmp_dir.cleanup()

    def test_compute_baseline_statistics(self):
        engine = ClimatologyEngine(historical_dataset=self.ds)
        clim_ds = engine.compute_baseline_statistics()

        self.assertIn("total_precipitation_clim_mean", clim_ds.data_vars)
        self.assertIn("total_precipitation_clim_std", clim_ds.data_vars)
        self.assertIn("total_precipitation_clim_median", clim_ds.data_vars)
        self.assertIn("total_precipitation_clim_p95", clim_ds.data_vars)

        # Check explicit honesty status flag
        self.assertIn("Development dataset is insufficient", clim_ds.attrs["climatology_status"])
        self.assertFalse(clim_ds.attrs["is_operational_30yr"])

    def test_climatology_mean_std_calculation(self):
        engine = ClimatologyEngine()
        clim_ds = engine.compute_baseline_statistics(ds=self.ds)
        temp_mean = float(clim_ds["2m_temperature_clim_mean"].values[0, 0])
        self.assertTrue(290.0 <= temp_mean <= 320.0)


if __name__ == "__main__":
    unittest.main()
