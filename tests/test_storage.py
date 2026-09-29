import unittest
import tempfile
import os
from pathlib import Path
from weather_core.storage.local_provider import LocalStorageProvider
from weather_core.storage.cloud_provider import S3StorageProvider
from weather_core.storage.factory import StorageFactory


class TestStorage(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.local_provider = LocalStorageProvider(root_dir=self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_local_storage_upload_download(self):
        test_file = Path(self.tmp_dir.name) / "sample.txt"
        test_file.write_text("Hello AERIS")

        uploaded = self.local_provider.upload_file(str(test_file), "raw/sample.txt")
        self.assertTrue(uploaded)
        self.assertTrue(self.local_provider.file_exists("raw/sample.txt"))

        dest_file = Path(self.tmp_dir.name) / "downloaded.txt"
        result_path = self.local_provider.download_file("raw/sample.txt", str(dest_file))
        self.assertEqual(Path(result_path).read_text(), "Hello AERIS")

    def test_cloud_storage_interface(self):
        s3 = S3StorageProvider(bucket_name="test-bucket")
        self.assertEqual(s3.bucket_name, "test-bucket")
        with self.assertRaises(NotImplementedError):
            s3.download_file("key.nc", "dest.nc")

    def test_storage_factory(self):
        local = StorageFactory.get_provider("local", data_root=self.tmp_dir.name)
        self.assertIsInstance(local, LocalStorageProvider)
        s3 = StorageFactory.get_provider("s3")
        self.assertIsInstance(s3, S3StorageProvider)


if __name__ == "__main__":
    unittest.main()
