"""
Configuration settings for the Heart Disease Predictor.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Dataset configuration
DATASET_UCI_ID = 45  # Heart Disease UCI dataset ID
DATASET_URL = "https://archive.ics.uci.edu/static/public/45/data.csv"
RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "heart_disease_processed.csv"

# Feature definitions
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET_COLUMN = "target"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Model configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model paths
MODEL_PATH = MODELS_DIR / "heart_disease_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
PIPELINE_PATH = MODELS_DIR / "full_pipeline.pkl"

# MLflow configuration
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT_NAME = "heart-disease-prediction"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
