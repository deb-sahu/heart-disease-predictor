"""
FastAPI application for Heart Disease Prediction API.
"""
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.schemas import PatientData, PredictionResponse, HealthResponse, ErrorResponse
from src.predict import HeartDiseasePredictor
from src.config import API_HOST, API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[HeartDiseasePredictor] = None


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
    except FileNotFoundError as e:
        logger.warning(f"Model not found: {e}. API will start but predictions will fail.")
        predictor = None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        predictor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Heart Disease Prediction API...")


# Create FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    A machine learning API for predicting heart disease risk based on patient health data.
    
    ## Features
    - Predict heart disease risk from patient health metrics
    - Returns probability scores and confidence levels
    - Built with MLOps best practices
    
    ## Model Information
    - Dataset: UCI Heart Disease Dataset
    - Features: 13 health indicators
    - Output: Binary classification (disease/no disease)
    """,
    version="1.0.0",
    lifespan=lifespan
)

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
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"model": PredictionResponse, "description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    }
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
    global predictor
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model is trained and available."
        )
    
    try:
        # Convert Pydantic model to dict
        input_data = patient_data.model_dump()
        
        # Log input (for monitoring)
        logger.info(f"Prediction request: age={input_data['age']}, sex={input_data['sex']}")
        
        # Make prediction
        result = predictor.predict(input_data)
        
        # Log prediction (for monitoring)
        logger.info(f"Prediction result: {result['prediction_label']} (confidence: {result['confidence']:.2%})")
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )
    
    return {
        "model_loaded": True,
        "features": predictor.feature_names,
        "feature_count": len(predictor.feature_names),
        "pipeline_path": str(predictor.pipeline_path)
    }


def start_server():
    """Start the API server."""
    uvicorn.run(
        "api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()

