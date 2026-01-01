"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Input schema for patient health data."""

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0 = female, 1 = male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0 = false, 1 = true)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0 = no, 1 = yes)")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)")
    thal: float = Field(
        ..., ge=0, le=7, description="Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: int = Field(..., description="Prediction (0 = no disease, 1 = disease)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability_no_disease: float = Field(..., ge=0, le=1, description="Probability of no heart disease")
    probability_disease: float = Field(..., ge=0, le=1, description="Probability of heart disease")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_label": "Heart Disease Present",
                "probability_no_disease": 0.23,
                "probability_disease": 0.77,
                "confidence": 0.77,
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
