import unittest
from weather_core.config.settings import Settings, get_settings


class TestConfig(unittest.TestCase):

    def test_settings_load_local(self):
        settings = Settings(config_path="configs/local.yaml")
        self.assertEqual(settings.environment, "local")
        self.assertEqual(settings.max_local_dataset_gb, 2.0)
        self.assertIn("D:/SIH26078_AERIS", settings.data_root)

    def test_settings_load_cloud(self):
        settings = Settings(config_path="configs/cloud.yaml")
        self.assertEqual(settings.environment, "cloud")
        self.assertEqual(settings.max_local_dataset_gb, 100.0)
        self.assertEqual(settings.provider_type, "s3")


if __name__ == "__main__":
    unittest.main()
