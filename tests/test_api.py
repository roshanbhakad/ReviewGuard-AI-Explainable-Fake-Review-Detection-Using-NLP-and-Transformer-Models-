"""Integration tests for Flask API endpoints."""

import unittest
import json
from app.web_server import create_app


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_predict_endpoint(self):
        payload = {
            "text": "BEST PRODUCT EVERRRR!!!! Changed my life forever!! MUST BUY 100%!",
            "rating": 5.0
        }
        res = self.client.post("/api/v1/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("fake_probability", data["data"])

    def test_explain_endpoint(self):
        payload = {
            "text": "Solid battery life, comfortable ear cups. A bit bulky in pocket.",
            "rating": 4.0
        }
        res = self.client.post("/api/v1/explain", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("highlighted_html", data["data"])
        self.assertIn("diagnostics", data["data"])
        self.assertIn("radar_data", data["data"])

    def test_batch_endpoint(self):
        payload = {
            "reviews": [
                {"id": 1, "text": "BEST EVER!! BUY NOW!!", "rating": 5.0},
                {"id": 2, "text": "Solid quality for the price.", "rating": 4.0}
            ]
        }
        res = self.client.post("/api/v1/batch", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["summary"]["total_reviews"], 2)

    def test_models_endpoint(self):
        res = self.client.get("/api/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("benchmark_metrics", data)

    def test_samples_endpoint(self):
        res = self.client.get("/api/v1/samples")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["samples"]), 0)


if __name__ == "__main__":
    unittest.main()
