import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from weather_core.preprocessing.standardizer import (
    CoordinateStandardizer,
    TimeStandardizer,
    UnitStandardizer,
    MissingDataHandler,
)
from weather_core.preprocessing.qc import QualityControlChecker
from weather_core.preprocessing.pipeline import WeatherPreprocessor
from scripts.create_sample_weather_data import create_sample_weather_dataset


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sample_nc = Path(self.tmp_dir.name) / "test_sample.nc"
        create_sample_weather_dataset(output_path=str(self.sample_nc))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_coordinate_standardization_and_sorting(self):
        # Create un-sorted dataset with non-standard coord names
        raw_ds = xr.Dataset(
            data_vars={"temp": (["lat", "lon"], np.array([[290.0, 295.0], [300.0, 305.0]]))},
            coords={
                "lat": [20.0, 10.0],  # Unsorted descending lat
                "lon": [85.0, 75.0],  # Unsorted descending lon
            },
        )
        std_ds = CoordinateStandardizer.standardize_coordinates(raw_ds)
        self.assertIn("latitude", std_ds.coords)
        self.assertIn("longitude", std_ds.coords)
        # Verify ascending sort order
        self.assertEqual(list(std_ds["latitude"].values), [10.0, 20.0])
        self.assertEqual(list(std_ds["longitude"].values), [75.0, 85.0])

    def test_time_standardization_and_deduplication(self):
        duplicate_times = pd.to_datetime(["2026-09-25 02:00", "2026-09-25 01:00", "2026-09-25 01:00"])
        raw_ds = xr.Dataset(
            data_vars={"precip": (["time"], [5.0, 10.0, 10.0])},
            coords={"time": duplicate_times},
        )
        std_ds = TimeStandardizer.standardize_time(raw_ds)
        self.assertEqual(len(std_ds["time"]), 2)  # Duplicate removed
        self.assertEqual(pd.Timestamp(std_ds["time"].values[0]), pd.Timestamp("2026-09-25 01:00:00"))  # Sorted ascending

    def test_explicit_unit_conversions(self):
        da_k = xr.DataArray(np.array([300.0]), attrs={"units": "K"})
        da_c = UnitStandardizer.convert_temperature(da_k, target_unit="degC")
        self.assertAlmostEqual(float(da_c.values[0]), 26.85, places=2)
        self.assertEqual(da_c.attrs["units"], "degC")

    def test_quality_control_bounds_checking(self):
        # Temp array with physical violation (100 K is below 180 K min)
        da_temp = xr.DataArray(
            np.array([300.0, 100.0, np.nan]),
            dims=["x"],
            attrs={"units": "K"},
        )
        qc_da, summary = QualityControlChecker.run_qc(da_temp, "2m_temperature")
        flags = qc_da.values
        self.assertEqual(flags[0], 0)  # Valid
        self.assertEqual(flags[1], 1)  # Out of bounds
        self.assertEqual(flags[2], 2)  # Missing / NaN
        self.assertFalse(summary["qc_passed"])
        self.assertEqual(summary["out_of_bounds_cells"], 1)

    def test_missing_data_interpolation(self):
        da_nan = xr.DataArray(
            np.array([[[10.0, np.nan, 30.0]]]),
            dims=["time", "latitude", "longitude"],
            coords={"time": [0], "latitude": [10.0], "longitude": [70.0, 71.0, 72.0]},
            attrs={"units": "mm"},
        )
        da_filled = MissingDataHandler.fill_missing_spatial_interpolation(da_nan, max_missing_pct=50.0)
        self.assertFalse(np.isnan(da_filled.values).any())
        self.assertAlmostEqual(float(da_filled.values[0, 0, 1]), 20.0, places=1)

    def test_full_preprocessing_pipeline(self):
        preprocessor = WeatherPreprocessor()
        ds_std, report = preprocessor.preprocess_dataset(
            source_path_or_ds=str(self.sample_nc),
            dataset_name="PIPELINE_TEST",
            target_temp_unit="degC",
        )
        self.assertTrue(report.pipeline_success)
        self.assertIn("2m_temperature_qc_flag", ds_std.data_vars)
        self.assertEqual(ds_std.attrs["preprocessing_status"], "PASSED_QUALITY_CONTROL")
        self.assertEqual(report.variables_summary["2m_temperature"]["units"], "degC")


if __name__ == "__main__":
    unittest.main()
