import unittest
import tempfile
from weather_core.data_manager.metadata import DatasetMetadata, GeographicalBounds
from weather_core.data_manager.dataset_manager import DatasetManager
from weather_core.storage.local_provider import LocalStorageProvider


class TestDatasetManager(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage = LocalStorageProvider(root_dir=self.tmp_dir.name)
        self.manager = DatasetManager(storage_provider=self.storage)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_metadata_dict_serialization(self):
        meta = DatasetMetadata(
            dataset_name="ERA5_SAMPLE",
            variable="total_precipitation",
            units="mm",
            source="ERA5",
            geographical_bounds=GeographicalBounds(6.0, 38.0, 68.0, 98.0),
            time_start="2026-09-01T00:00:00Z",
            time_end="2026-09-05T00:00:00Z",
            resolution_km=12.0,
            local_or_cloud="local",
            file_location="data/samples/era5_sample.nc",
            size_bytes=1048576,
        )

        meta_dict = meta.to_dict()
        self.assertEqual(meta_dict["dataset_name"], "ERA5_SAMPLE")
        self.assertEqual(meta_dict["geographical_bounds"]["min_lat"], 6.0)

        restored = DatasetMetadata.from_dict(meta_dict)
        self.assertEqual(restored.dataset_name, meta.dataset_name)
        self.assertEqual(restored.geographical_bounds.max_lon, 98.0)


if __name__ == "__main__":
    unittest.main()
