import unittest
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr

from weather_core.ingestion.local_netcdf import LocalNetCDFProvider
from weather_core.ingestion.validator import DatasetValidator, DatasetValidationError
from weather_core.ingestion.nwm_providers import ERA5Provider, NCUMProvider
from weather_core.data_manager import DataSizeExceededError
from scripts.create_sample_weather_data import create_sample_weather_dataset


class TestIngestion(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sample_nc = Path(self.tmp_dir.name) / "test_sample.nc"
        create_sample_weather_dataset(output_path=str(self.sample_nc))
        self.provider = LocalNetCDFProvider()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_open_and_validate_valid_dataset(self):
        ds = self.provider.open_dataset(str(self.sample_nc))
        self.assertIn("total_precipitation", ds.data_vars)
        self.assertIn("2m_temperature", ds.data_vars)
        self.assertIn("10m_wind_speed", ds.data_vars)

        report = self.provider.validate(str(self.sample_nc))
        self.assertTrue(report["valid"])
        self.assertEqual(report["latitude_coord"], "latitude")
        self.assertEqual(report["longitude_coord"], "longitude")
        self.assertEqual(report["lat_range"], [8.0, 36.0])

    def test_metadata_extraction(self):
        metadata = self.provider.extract_metadata(str(self.sample_nc), dataset_name="INDIA_TEST")
        self.assertEqual(metadata.dataset_name, "INDIA_TEST")
        self.assertEqual(metadata.source, "LocalNetCDF")
        self.assertEqual(metadata.geographical_bounds.min_lat, 8.0)
        self.assertEqual(metadata.geographical_bounds.max_lat, 36.0)

    def test_field_extraction_with_cropping(self):
        field = self.provider.extract_field(
            str(self.sample_nc),
            variable_name="total_precipitation",
            time_index=0,
            spatial_bounds={"min_lat": 18.0, "max_lat": 24.0, "min_lon": 80.0, "max_lon": 90.0},
        )
        self.assertEqual(field.name, "total_precipitation")
        self.assertTrue(float(field.coords["latitude"].min()) >= 18.0)
        self.assertTrue(float(field.coords["latitude"].max()) <= 24.0)

    def test_validation_rejects_missing_coordinates(self):
        invalid_ds = xr.Dataset(
            data_vars={"temp": (["x", "y"], np.ones((5, 5)))},
            attrs={"units": "K"},
        )
        with self.assertRaises(DatasetValidationError):
            DatasetValidator.validate_dataset(invalid_ds)

    def test_validation_rejects_invalid_units(self):
        invalid_units_ds = xr.Dataset(
            data_vars={"total_precipitation": (["time", "latitude", "longitude"], np.zeros((2, 2, 2)))},
            coords={
                "time": [0, 1],
                "latitude": [10.0, 11.0],
                "longitude": [70.0, 71.0],
            },
        )
        invalid_units_ds["total_precipitation"].attrs["units"] = "invalid_speed_unit"
        with self.assertRaises(DatasetValidationError):
            DatasetValidator.validate_dataset(invalid_units_ds)

    def test_nwm_provider_size_guardrails(self):
        era5 = ERA5Provider()
        # Request 10 GB (exceeds 2.0 GB limit)
        with self.assertRaises(DataSizeExceededError):
            era5.request_subset("tp", "2026-01-01", "2026-01-05", {}, estimated_size_gb=10.0)


if __name__ == "__main__":
    unittest.main()
