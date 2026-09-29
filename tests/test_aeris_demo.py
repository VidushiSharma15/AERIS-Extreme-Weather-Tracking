import unittest
import os
import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure root workspace is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.run_aeris_demo import run_aeris_demo_pipeline
from backend.main import app


class TestAERISDemoPipeline(unittest.TestCase):
    """
    Test suite for Phase 14 End-to-End Integration, Demo Pipeline, and API Endpoints.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.manifest_path = Path("D:/SIH26078_AERIS/data/processed/demo/aeris_demo_manifest.json")

    def test_01_run_aeris_demo_pipeline(self):
        """Verify end-to-end execution of run_aeris_demo_pipeline()."""
        manifest = run_aeris_demo_pipeline()

        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["demo_id"], "AERIS_DEMO_CYCLONE_AMPHAN_2020")
        self.assertEqual(manifest["system_status"]["overall"], "COMPLETE")
        self.assertTrue(self.manifest_path.exists())

    def test_02_api_demo_manifest_endpoint(self):
        """Test GET /api/v1/demo/manifest endpoint."""
        response = self.client.get("/api/v1/demo/manifest")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["demo_id"], "AERIS_DEMO_CYCLONE_AMPHAN_2020")
        self.assertIn("dataset_metadata", data)
        self.assertIn("spherical_gnn", data)

    def test_03_api_demo_status_endpoint(self):
        """Test GET /api/v1/demo/status endpoint."""
        response = self.client.get("/api/v1/demo/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["overall_status"], "COMPLETE")
        self.assertEqual(data["data_ingestion"], "PASS")
        self.assertEqual(data["spherical_gnn"], "PASS")
        self.assertEqual(data["spatio_temporal_tracking"], "PASS")
        self.assertEqual(data["conditional_diffusion_downscaling"], "PASS")


if __name__ == "__main__":
    unittest.main()
