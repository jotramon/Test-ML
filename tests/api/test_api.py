import os
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Set required env vars before importing the app
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCP_REGION", "us-central1")
os.environ.setdefault("GCP_BUCKET_ARTIFACTS", "test-project-mle-artifacts")
os.environ.setdefault("BQ_DATASET", "mle_challenge")

from challenge import app


class TestHealth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class TestPipelineTrain(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=True)

    def _mock_bq_count(self, n: int):
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: n if key == "n" else None
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([mock_row])
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = mock_result
        return mock_client

    def test_train_returns_422_when_raw_flights_empty(self):
        bq_mock = self._mock_bq_count(0)
        with patch("challenge.api.api._check_raw_flights", return_value=0):
            response = self.client.post("/pipeline/train")
        self.assertEqual(response.status_code, 422)
        self.assertIn("empty", response.json()["detail"])

    def test_train_returns_500_on_pipeline_failure(self):
        with patch("challenge.api.api._check_raw_flights", return_value=100), \
             patch("challenge.api.api._submit_and_wait", side_effect=RuntimeError("failed")):
            response = self.client.post("/pipeline/train")
        self.assertEqual(response.status_code, 500)
        self.assertIn("failed", response.json()["detail"]["details"])

    def test_train_returns_200_on_success(self):
        mock_job = MagicMock()
        mock_job.name = "projects/test/locations/us-central1/pipelineJobs/123"

        with patch("challenge.api.api._check_raw_flights", return_value=1000), \
             patch("challenge.api.api._submit_and_wait", return_value=mock_job):
            response = self.client.post("/pipeline/train")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_display_name"], "delay-classifier")
        self.assertIn("pipeline_job_id", body)
        self.assertIn("metrics", body)


class TestPipelinePredict(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_predict_returns_422_when_no_model(self):
        with patch("challenge.api.api._check_model_exists", return_value=False):
            response = self.client.post("/pipeline/predict")
        self.assertEqual(response.status_code, 422)
        self.assertIn("no trained model", response.json()["detail"])

    def test_predict_returns_500_on_pipeline_failure(self):
        with patch("challenge.api.api._check_model_exists", return_value=True), \
             patch("challenge.api.api._submit_and_wait", side_effect=RuntimeError("failed")):
            response = self.client.post("/pipeline/predict")
        self.assertEqual(response.status_code, 500)
        self.assertIn("failed", response.json()["detail"]["details"])

    def test_predict_returns_200_on_success(self):
        mock_job = MagicMock()
        mock_job.name = "projects/test/locations/us-central1/pipelineJobs/456"
        mock_predictions = {
            "total_predictions": 68206,
            "predictions": [
                {"OPERA": "Grupo LATAM", "TIPOVUELO": "I", "MES": 6, "predicted_delay": 0}
            ],
        }

        with patch("challenge.api.api._check_model_exists", return_value=True), \
             patch("challenge.api.api._submit_and_wait", return_value=mock_job), \
             patch("challenge.api.api._query_predictions", return_value=mock_predictions):
            response = self.client.post("/pipeline/predict")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_predictions"], 68206)
        self.assertIn("pipeline_job_id", body)


class TestPredictResults(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_results_returns_200_with_empty_predictions(self):
        mock_result = {"total_predictions": 0, "predictions": []}
        with patch("challenge.api.api._query_predictions", return_value=mock_result):
            response = self.client.get("/pipeline/predict/results")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_predictions"], 0)

    def test_results_respects_pagination(self):
        """GET /results echoes page and page_size in the response body."""
        mock_result = {
            "total_predictions": 100,
            "predictions": [
                {
                    "OPERA": "Grupo LATAM",
                    "TIPOVUELO": "I",
                    "MES": (i % 12) + 1,
                    "predicted_delay": 0,
                }
                for i in range(5)
            ],
        }
        with patch("challenge.api.api._query_predictions", return_value=mock_result) as mock_q:
            response = self.client.get("/pipeline/predict/results?page=2&page_size=5")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["page"], 2)
        self.assertEqual(body["page_size"], 5)


class TestPredictResultsFilters(unittest.TestCase):
    """Tests for filter and pagination query parameters on GET /pipeline/predict/results."""

    def setUp(self):
        """Set up the test client."""
        self.client = TestClient(app, raise_server_exceptions=True)
        self._empty_result = {"total_predictions": 0, "predictions": []}

    def test_results_opera_filter_passed_to_query(self):
        """opera query param is split and forwarded to _query_predictions as a list."""
        with patch(
            "challenge.api.api._query_predictions", return_value=self._empty_result
        ) as mock_q:
            self.client.get("/pipeline/predict/results?opera=Grupo+LATAM,Sky+Airline")

        call_kwargs = mock_q.call_args[0]
        opera_arg = call_kwargs[2]
        self.assertEqual(opera_arg, ["Grupo LATAM", "Sky Airline"])

    def test_results_tipovuelo_filter_passed_to_query(self):
        """tipovuelo query param is split and forwarded to _query_predictions as a list."""
        with patch(
            "challenge.api.api._query_predictions", return_value=self._empty_result
        ) as mock_q:
            self.client.get("/pipeline/predict/results?tipovuelo=I")

        call_kwargs = mock_q.call_args[0]
        tipovuelo_arg = call_kwargs[3]
        self.assertEqual(tipovuelo_arg, ["I"])

    def test_results_mes_filter_converts_strings_to_integers(self):
        """mes query param is parsed into a list of ints and forwarded to _query_predictions."""
        with patch(
            "challenge.api.api._query_predictions", return_value=self._empty_result
        ) as mock_q:
            self.client.get("/pipeline/predict/results?mes=3,6,12")

        call_kwargs = mock_q.call_args[0]
        mes_arg = call_kwargs[4]
        self.assertEqual(mes_arg, [3, 6, 12])

    def test_results_invalid_mes_returns_422(self):
        """Non-integer mes value returns 422 with an informative detail message."""
        with patch("challenge.api.api._query_predictions", return_value=self._empty_result):
            response = self.client.get("/pipeline/predict/results?mes=not_a_number")
        self.assertEqual(response.status_code, 422)
        self.assertIn("mes must be", response.json()["detail"])

    def test_results_page_and_page_size_passed_to_query(self):
        """page and page_size query params are forwarded to _query_predictions correctly."""
        mock_result = {"total_predictions": 0, "predictions": []}
        with patch(
            "challenge.api.api._query_predictions", return_value=mock_result
        ) as mock_q:
            response = self.client.get("/pipeline/predict/results?page=3&page_size=25")

        self.assertEqual(response.status_code, 200)
        call_kwargs = mock_q.call_args[0]
        self.assertEqual(call_kwargs[0], 3)
        self.assertEqual(call_kwargs[1], 25)
        body = response.json()
        self.assertEqual(body["page"], 3)
        self.assertEqual(body["page_size"], 25)
