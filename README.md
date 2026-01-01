# Heart Disease Predictor - MLOps Project

[![CI/CD Pipeline](https://github.com/deb-sahu/heart-disease-predictor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/deb-sahu/heart-disease-predictor/actions/workflows/ci-cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready machine learning solution for predicting heart disease risk, built with modern MLOps best practices including experiment tracking, CI/CD pipelines, containerization, and API deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Docker Deployment](#docker-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Model Information](#model-information)
- [Architecture](#architecture)

---

## 🎯 Overview

This project implements an end-to-end ML pipeline for predicting heart disease risk based on patient health data from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

### Problem Statement
Build a machine learning classifier to predict the risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

### Dataset
- **Source**: UCI Machine Learning Repository
- **Samples**: 303 patients
- **Features**: 13 health indicators (age, sex, blood pressure, cholesterol, etc.)
- **Target**: Binary classification (presence/absence of heart disease)

---

## ✨ Features

- **Data Pipeline**: Automated data acquisition, preprocessing, and feature engineering
- **Model Training**: Multiple classifiers with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for logging parameters, metrics, and artifacts
- **REST API**: FastAPI-based prediction endpoint with input validation
- **Containerization**: Docker support for reproducible deployments
- **CI/CD**: GitHub Actions pipeline with linting, testing, and deployment
- **Testing**: Comprehensive unit and integration tests

---

## 📁 Project Structure

```
heart-disease-predictor/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── api/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   └── schemas.py              # Pydantic request/response schemas
├── data/                       # Dataset storage (gitignored)
├── models/                     # Trained model artifacts (gitignored)
├── artifacts/                  # Training artifacts (plots, reports)
├── scripts/
│   ├── download_data.py        # Dataset download script
│   └── run_training.py         # Training runner script
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration settings
│   ├── data_loader.py          # Data acquisition utilities
│   ├── preprocessing.py        # Feature engineering pipeline
│   ├── train.py                # Model training with MLflow
│   └── predict.py              # Prediction utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Pytest fixtures
│   ├── test_data_loader.py     # Data loader tests
│   ├── test_preprocessing.py   # Preprocessing tests
│   ├── test_model.py           # Model training tests
│   └── test_api.py             # API endpoint tests
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-container setup
├── Makefile                    # Convenience commands
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
├── pytest.ini                  # Test configuration
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (recommended: 3.11)
- pip or conda
- Docker (optional, for containerization)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# or: pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Download dataset and train model
make train
# or: python -m src.train
```

### 3. Run the API

```bash
# Start the API server
make run
# or: uvicorn api.app:app --reload --port 8000
```

### 4. Test the Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

---

## 📦 Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using Make

```bash
make install      # Production dependencies
make install-dev  # All dependencies including dev tools
```

### Dependencies

| Package | Purpose |
|---------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | ML models and preprocessing |
| mlflow | Experiment tracking |
| fastapi, uvicorn | REST API |
| pytest | Testing framework |
| flake8, black | Code quality |

---

## 🔧 Usage

### Training Models

```bash
# Full training with MLflow tracking
python -m src.train

# Or using Make
make train
```

Training will:
1. Download the dataset (if not present)
2. Preprocess and split data
3. Train Logistic Regression and Random Forest models
4. Log experiments to MLflow
5. Save the best model to `models/`

### Making Predictions

#### Python API

```python
from src.predict import HeartDiseasePredictor

predictor = HeartDiseasePredictor().load()
result = predictor.predict({
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
})
print(result)
# {'prediction': 1, 'prediction_label': 'Heart Disease Present', 
#  'probability_disease': 0.77, 'confidence': 0.77}
```

#### REST API

See [API Documentation](#api-documentation) below.

---

## 📡 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information and health |
| `/health` | GET | Health check for orchestration |
| `/predict` | POST | Make a prediction |
| `/model/info` | GET | Model metadata |

### Interactive Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Request/Response Examples

#### Health Check
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```
```json
{
  "prediction": 1,
  "prediction_label": "Heart Disease Present",
  "probability_no_disease": 0.23,
  "probability_disease": 0.77,
  "confidence": 0.77
}
```

### Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| age | int | 0-120 | Age in years |
| sex | int | 0-1 | Sex (0=female, 1=male) |
| cp | int | 0-3 | Chest pain type |
| trestbps | int | 0-300 | Resting blood pressure (mm Hg) |
| chol | int | 0-600 | Serum cholesterol (mg/dl) |
| fbs | int | 0-1 | Fasting blood sugar > 120 mg/dl |
| restecg | int | 0-2 | Resting ECG results |
| thalach | int | 0-250 | Maximum heart rate achieved |
| exang | int | 0-1 | Exercise induced angina |
| oldpeak | float | 0-10 | ST depression induced by exercise |
| slope | int | 0-2 | Slope of peak exercise ST segment |
| ca | float | 0-4 | Number of major vessels (fluoroscopy) |
| thal | float | 0-7 | Thalassemia type |

---

## 📊 MLflow Experiment Tracking

### Starting MLflow UI

```bash
make mlflow-ui
# or: mlflow ui --port 5000
```

Visit http://localhost:5000 to view experiments.

### Tracked Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Cross-validation scores**: 5-fold CV metrics

### Logged Artifacts

- Confusion matrix plots
- ROC curve plots
- Classification reports
- Trained model files

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
make docker-build
# or: docker build -t heart-disease-predictor .

# Run the container
make docker-run
# or: docker run -d -p 8000:8000 --name heart-disease-api heart-disease-predictor

# Test the container
make docker-test
```

### Docker Compose (with MLflow)

```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000

# Stop services
docker-compose down
```

### Container Features

- Multi-stage build for optimized image size
- Non-root user for security
- Health checks for orchestration
- Volume mounts for model updates

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) includes:

### Pipeline Stages

1. **Lint**: Code quality checks with flake8, black, isort
2. **Test**: Unit tests with pytest and coverage reporting
3. **Train**: Model training with artifact generation
4. **Docker Build**: Container image creation
5. **Integration Tests**: End-to-end API testing

### Triggers

- Push to `main`, `master`, or `develop` branches
- Pull requests to `main` or `master`

### Artifacts

- Test coverage reports
- Model artifacts
- MLflow experiment logs
- Docker images

---

## 🧪 Testing

### Run All Tests

```bash
make test
# or: pytest tests/ -v
```

### Run with Coverage

```bash
make test-cov
# or: pytest tests/ --cov=src --cov=api --cov-report=html
```

### Test Categories

- `tests/test_data_loader.py`: Data loading and validation
- `tests/test_preprocessing.py`: Feature engineering pipeline
- `tests/test_model.py`: Model training and evaluation
- `tests/test_api.py`: API endpoint testing

---

## 🤖 Model Information

### Models Evaluated

| Model | Accuracy | ROC-AUC | CV ROC-AUC |
|-------|----------|---------|------------|
| Logistic Regression | 0.87 | 0.96 | 0.90 |
| Random Forest | 0.89 | 0.95 | 0.89 |

### Selected Model

**Logistic Regression** was selected for production due to:
- Higher cross-validation stability
- Better generalization (ROC-AUC: 0.96)
- Simpler deployment and interpretability
- Comparable accuracy to Random Forest

### Feature Importance

The most predictive features for heart disease:
1. `thal` (Thalassemia type)
2. `ca` (Number of major vessels)
3. `cp` (Chest pain type)
4. `oldpeak` (ST depression)
5. `exang` (Exercise-induced angina)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐ │
│  │  Lint   │→ │  Test   │→ │  Train  │→ │  Docker Build/Push  │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Training Pipeline                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Data Loader  │ → │ Preprocessor │ → │ Model Training       │ │
│  │ (UCI Repo)   │   │ (Sklearn)    │   │ (MLflow Tracking)    │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Serving Layer                               │
│  ┌──────────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ FastAPI Endpoint │ ← │ Model Loader │ ← │ Saved Pipeline  │  │
│  │ /predict         │   │              │   │ (joblib)        │  │
│  └──────────────────┘   └──────────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Container                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Python 3.11 Runtime                      │ │
│  │  ┌─────────┐  ┌───────────────┐  ┌──────────────────────┐  │ │
│  │  │ Uvicorn │→ │ FastAPI App   │→ │ ML Pipeline          │  │ │
│  │  │ :8000   │  │ (Validation)  │  │ (Preprocessing+Model)│  │ │
│  │  └─────────┘  └───────────────┘  └──────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
