import unittest
from fastapi.testclient import TestClient
from backend.main import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "AERIS")

    def test_api_v1_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_get_events_endpoint(self):
        response = self.client.get("/api/v1/events")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(isinstance(data, list) or "events" in data)

    def test_get_event_by_invalid_id_returns_404(self):
        response = self.client.get("/api/v1/events/INVALID_EVENT_ID_999")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertTrue(data["error"])
        self.assertEqual(data["status_code"], 404)

    def test_get_alerts_endpoint(self):
        response = self.client.get("/api/v1/alerts")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_get_data_status_endpoint(self):
        response = self.client.get("/api/v1/data/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("provider_type", data)

    def test_get_metadata_endpoint(self):
        response = self.client.get("/api/v1/metadata")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dataset_name", data)
        self.assertIn("processing_version", data)


if __name__ == "__main__":
    unittest.main()
