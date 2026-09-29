import unittest
import tempfile
from weather_core.data_manager.dataset_manager import DatasetManager, DataSizeExceededError
from weather_core.data_manager.metadata import DatasetMetadata, GeographicalBounds
from weather_core.storage.local_provider import LocalStorageProvider


class TestSafetyGuardrails(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage = LocalStorageProvider(root_dir=self.tmp_dir.name)
        self.manager = DatasetManager(storage_provider=self.storage)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_small_dataset_passes_safety_check(self):
        # 50 MB dataset (well below 2.0 GB limit)
        size_50mb = 50 * 1024 * 1024
        self.assertTrue(self.manager.check_size_safety(size_50mb))

    def test_oversized_dataset_raises_error(self):
        # 5.0 GB dataset (exceeds 2.0 GB limit)
        size_5gb = 5 * 1024 * 1024 * 1024
        with self.assertRaises(DataSizeExceededError):
            self.manager.check_size_safety(size_5gb)

    def test_register_oversized_local_dataset_blocked(self):
        meta = DatasetMetadata(
            dataset_name="HUGE_GLOBAL_ERA5",
            variable="total_precipitation",
            units="mm",
            source="ERA5",
            geographical_bounds=GeographicalBounds(-90.0, 90.0, -180.0, 180.0),
            time_start="1990-01-01T00:00:00Z",
            time_end="2026-01-01T00:00:00Z",
            resolution_km=12.0,
            local_or_cloud="local",
            file_location="data/raw/huge.nc",
            size_bytes=50 * 1024 * 1024 * 1024,  # 50 GB
        )

        with self.assertRaises(DataSizeExceededError):
            self.manager.register_dataset(meta)


if __name__ == "__main__":
    unittest.main()
