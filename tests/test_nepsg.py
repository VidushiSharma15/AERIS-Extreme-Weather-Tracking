import unittest
from pathlib import Path
from scripts.download_nepsg_tigge import build_ecds_request
from weather_core.ingestion import NEPSGProvider


class TestNEPSGECDSMigration(unittest.TestCase):
    def test_ecds_request_payload_structure(self):
        req = build_ecds_request()
        self.assertEqual(req["dataset"], "tigge-forecasts")
        self.assertEqual(req["origin"], "dems")
        self.assertEqual(req["year"], "2020")
        self.assertEqual(req["month"], "05")
        self.assertEqual(req["day"], "17")
        self.assertEqual(req["time"], "00:00")
        self.assertEqual(len(req["leadtime_hour"]), 13)
        self.assertIn("total_precipitation", req["variable"])
        self.assertIn("10_m_u_component_of_wind", req["variable"])
        self.assertIn("10_m_v_component_of_wind", req["variable"])
        self.assertIn("mean_sea_level_pressure", req["variable"])
        self.assertEqual(req["area"], [25.0, 80.0, 10.0, 95.0])
        self.assertEqual(req["data_format"], "grib")

    def test_nepsg_provider_metadata(self):
        provider = NEPSGProvider()
        self.assertEqual(provider.provider_name, "NEPS-G")
        self.assertEqual(provider.provider_wmo_code, "dems")
        self.assertEqual(provider.data_source_type, "NWP_FORECAST")
        self.assertEqual(provider.ensemble_members, list(range(12)))


if __name__ == "__main__":
    unittest.main()
