"""
FastAPI application for Heart Disease Prediction API.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import ErrorResponse, HealthResponse, PatientData, PredictionResponse
from src.config import API_HOST, API_PORT
from src.predict import HeartDiseasePredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[HeartDiseasePredictor] = None

# ============================================================================
# Prometheus Custom Metrics
# ============================================================================

# Prediction metrics
PREDICTIONS_TOTAL = Counter(
    "heart_disease_predictions_total",
    "Total number of predictions made",
    ["prediction_result"]
)

PREDICTION_LATENCY = Histogram(
    "heart_disease_prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTION_CONFIDENCE = Histogram(
    "heart_disease_prediction_confidence",
    "Confidence scores of predictions",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

MODEL_LOADED = Gauge(
    "heart_disease_model_loaded",
    "Whether the ML model is currently loaded (1=loaded, 0=not loaded)"
)

PREDICTION_ERRORS = Counter(
    "heart_disease_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global predictor

    # Startup
    logger.info("Starting Heart Disease Prediction API...")
    try:
        predictor = HeartDiseasePredictor()
        predictor.load()
        logger.info("Model loaded successfully")
        MODEL_LOADED.set(1)
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}. API will start but predictions will fail.")
        predictor = None
        MODEL_LOADED.set(0)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        predictor = None
        MODEL_LOADED.set(0)

    yield

    # Shutdown
    logger.info("Shutting down Heart Disease Prediction API...")
    MODEL_LOADED.set(0)


# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    A machine learning API for predicting heart disease risk based on patient health data.

    ## Features
    - Predict heart disease risk from patient health metrics
    - Returns probability scores and confidence levels
    - Built with MLOps best practices
    - Prometheus metrics at /metrics endpoint

    ## Model Information
    - Dataset: UCI Heart Disease Dataset
    - Features: 13 health indicators
    - Output: Binary classification (disease/no disease)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Add Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,  # Always enable metrics
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    inprogress_name="heart_disease_inprogress_requests",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response
    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} - Duration: {duration:.3f}s")

    return response


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information."""
    return HealthResponse(status="healthy", model_loaded=predictor is not None, version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    return HealthResponse(status="healthy", model_loaded=predictor is not None, version="1.0.0")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict(patient_data: PatientData):
    """
    Predict heart disease risk for a patient.

    - **age**: Age in years
    - **sex**: Sex (0 = female, 1 = male)
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl
    - **restecg**: Resting ECG results
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina
    - **oldpeak**: ST depression induced by exercise
    - **slope**: Slope of peak exercise ST segment
    - **ca**: Number of major vessels (0-4)
    - **thal**: Thalassemia type
    """
    if predictor is None:
        PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please ensure the model is trained and available."
        )

    start_time = datetime.now()

    try:
        # Convert Pydantic model to dict
        input_data = patient_data.model_dump()

        # Log input (for monitoring)
        logger.info(f"Prediction request: age={input_data['age']}, sex={input_data['sex']}")

        # Make prediction
        result = predictor.predict(input_data)

        # Record metrics
        prediction_label = "disease" if result["prediction"] == 1 else "no_disease"
        PREDICTIONS_TOTAL.labels(prediction_result=prediction_label).inc()
        PREDICTION_CONFIDENCE.observe(result["confidence"])

        # Record latency
        latency = (datetime.now() - start_time).total_seconds()
        PREDICTION_LATENCY.observe(latency)

        # Log prediction (for monitoring)
        logger.info(f"Prediction result: {result['prediction_label']} (confidence: {result['confidence']:.2%})")

        return PredictionResponse(**result)

    except ValueError as e:
        PREDICTION_ERRORS.labels(error_type="validation_error").inc()
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="internal_error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    return {
        "model_loaded": True,
        "features": predictor.feature_names,
        "feature_count": len(predictor.feature_names),
        "pipeline_path": str(predictor.pipeline_path),
    }


def start_server():
    """Start the API server."""
    uvicorn.run("api.app:app", host=API_HOST, port=API_PORT, reload=False, log_level="info")


if __name__ == "__main__":
    start_server()
