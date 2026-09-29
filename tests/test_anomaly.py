import unittest
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr

from weather_core.anomaly.engine import AnomalyEngine
from weather_core.anomaly.severity import SeverityEngine, SeverityLevel
from weather_core.anomaly.spatial_regions import SpatialRegionDetector, ExtremeWeatherEvent
from weather_core.anomaly.geojson import GeoJSONExporter
from weather_core.anomaly.ensemble import EnsembleProcessor
from weather_core.anomaly.efi import EFICalculator
from weather_core.preprocessing import WeatherPreprocessor
from scripts.create_sample_weather_data import create_sample_weather_dataset


class TestAnomaly(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sample_nc = Path(self.tmp_dir.name) / "test_sample.nc"
        create_sample_weather_dataset(output_path=str(self.sample_nc))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_zscore_calculation_and_zero_std_handling(self):
        # Create dataset with constant field (std = 0)
        ds_constant = xr.Dataset(
            data_vars={"temp": (["time", "latitude", "longitude"], np.ones((3, 5, 5)) * 300.0)},
            coords={
                "time": [0, 1, 2],
                "latitude": np.linspace(10, 14, 5),
                "longitude": np.linspace(70, 74, 5),
            },
        )
        engine = AnomalyEngine()
        ds_anom = engine.compute_anomaly_fields(ds_constant)
        z_vals = ds_anom["temp_zscore"].values
        self.assertFalse(np.isnan(z_vals).any())  # Zero std handling prevented NaN/Inf
        self.assertTrue(np.all(z_vals == 0.0))

    def test_severity_classification(self):
        severity = SeverityEngine()
        self.assertEqual(severity.classify_z_score(1.5), SeverityLevel.NORMAL)
        self.assertEqual(severity.classify_z_score(2.5), SeverityLevel.WATCH)
        self.assertEqual(severity.classify_z_score(3.5), SeverityLevel.MODERATE)
        self.assertEqual(severity.classify_z_score(5.0), SeverityLevel.SEVERE)

    def test_connected_components_region_detection(self):
        preprocessor = WeatherPreprocessor()
        ds_std, _ = preprocessor.preprocess_dataset(str(self.sample_nc))
        engine = AnomalyEngine()
        ds_anom, events = engine.detect_extreme_events(ds_std, source_dataset_name="TEST")

        self.assertGreater(len(events), 0)
        first_evt = events[0]
        self.assertIsInstance(first_evt, ExtremeWeatherEvent)
        self.assertGreater(first_evt.area_km2, 0.0)
        self.assertGreater(first_evt.affected_grid_cells, 0)
        self.assertTrue(first_evt.centroid.latitude >= 8.0)
        self.assertTrue(first_evt.centroid.longitude >= 68.0)

    def test_geojson_export(self):
        preprocessor = WeatherPreprocessor()
        ds_std, _ = preprocessor.preprocess_dataset(str(self.sample_nc))
        engine = AnomalyEngine()
        _, events = engine.detect_extreme_events(ds_std)

        geojson_data = GeoJSONExporter.events_to_geojson(events)
        self.assertEqual(geojson_data["type"], "FeatureCollection")
        self.assertEqual(len(geojson_data["features"]), len(events))
        first_feature = geojson_data["features"][0]
        self.assertEqual(first_feature["geometry"]["type"], "Polygon")
        self.assertIn("severity", first_feature["properties"])

    def test_ensemble_and_efi_interfaces(self):
        ens_res = EnsembleProcessor.compute_ensemble_stats(None)
        self.assertIn("awaiting NEPS-G", ens_res["status"])

        efi_res = EFICalculator.compute_efi_prototype(None, None, "tp")
        self.assertFalse(efi_res["is_full_efi"])
        self.assertIn("NEPS-G", efi_res["status"])

    def test_end_to_end_anomaly_pipeline(self):
        # REAL WEATHER DATA -> PREPROCESSING -> BASELINE -> ANOMALY -> EXTREME REGIONS -> GEOJSON
        preprocessor = WeatherPreprocessor()
        ds_std, _ = preprocessor.preprocess_dataset(str(self.sample_nc))

        engine = AnomalyEngine()
        ds_anom, events = engine.detect_extreme_events(ds_std)

        out_json = Path(self.tmp_dir.name) / "test_out.geojson"
        GeoJSONExporter.save_geojson(events, str(out_json))

        self.assertTrue(out_json.exists())
        self.assertGreater(out_json.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
