"""
Unit tests for the FastAPI application.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor."""
        mock = MagicMock()
        mock.predict.return_value = {
            "prediction": 1,
            "prediction_label": "Heart Disease Present",
            "probability_no_disease": 0.23,
            "probability_disease": 0.77,
            "confidence": 0.77,
        }
        mock.feature_names = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ]
        mock.pipeline_path = Path("/mock/path/pipeline.pkl")
        return mock

    @pytest.fixture
    def client(self, mock_predictor):
        """Create test client with mocked predictor."""
        with patch("api.app.predictor", mock_predictor):
            from api.app import app

            with TestClient(app) as client:
                yield client

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_predict_endpoint_valid_input(self, client, sample_patient_data):
        """Test predict endpoint with valid input."""
        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "prediction_label" in data
        assert "probability_disease" in data
        assert "confidence" in data

    def test_predict_endpoint_missing_field(self, client, sample_patient_data):
        """Test predict endpoint with missing required field."""
        del sample_patient_data["age"]

        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_value(self, client, sample_patient_data):
        """Test predict endpoint with invalid field value."""
        sample_patient_data["age"] = -5  # Invalid age

        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_invalid_type(self, client, sample_patient_data):
        """Test predict endpoint with invalid field type."""
        sample_patient_data["age"] = "not_a_number"

        response = client.post("/predict", json=sample_patient_data)
        assert response.status_code == 422  # Validation error


class TestAPISchemaValidation:
    """Tests for API schema validation."""

    def test_patient_data_schema_valid(self, sample_patient_data):
        """Test PatientData schema with valid data."""
        from api.schemas import PatientData

        # Should not raise
        patient = PatientData(**sample_patient_data)
        assert patient.age == 63
        assert patient.sex == 1

    def test_patient_data_schema_age_bounds(self):
        """Test PatientData age validation."""
        from api.schemas import PatientData

        valid_data = {
            "age": 50,
            "sex": 1,
            "cp": 0,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 2,
        }

        # Valid age
        PatientData(**{**valid_data, "age": 0})
        PatientData(**{**valid_data, "age": 120})

        # Invalid ages
        with pytest.raises(ValueError):
            PatientData(**{**valid_data, "age": -1})
        with pytest.raises(ValueError):
            PatientData(**{**valid_data, "age": 121})

    def test_prediction_response_schema(self):
        """Test PredictionResponse schema."""
        from api.schemas import PredictionResponse

        response = PredictionResponse(
            prediction=1,
            prediction_label="Heart Disease Present",
            probability_no_disease=0.23,
            probability_disease=0.77,
            confidence=0.77,
        )

        assert response.prediction == 1
        assert response.confidence == 0.77
