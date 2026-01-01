# Technical Guide: Heart Disease Predictor MLOps Pipeline

This document provides a detailed explanation of the MLflow integration, model training workflow, sklearn pipeline architecture, and Docker containerization.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [MLflow Integration](#2-mlflow-integration)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Training Workflow](#4-model-training-workflow)
5. [Sklearn Pipeline Architecture](#5-sklearn-pipeline-architecture)
6. [API Design](#6-api-design)
7. [Docker Configuration](#7-docker-configuration)
8. [CI/CD Pipeline](#8-cicd-pipeline)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────────┐ │
│  │ UCI Dataset │ →  │ data_loader  │ →  │ data/heart_disease_raw.csv      │ │
│  │ (Remote)    │    │ .py          │    │ (Local Cache)                   │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PREPROCESSING LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    ColumnTransformer (preprocessor)                      ││
│  │  ┌─────────────────────────────┐  ┌────────────────────────────────────┐││
│  │  │     Numeric Pipeline        │  │      Categorical Pipeline          │││
│  │  │  ┌───────────────────────┐  │  │  ┌──────────────────────────────┐  │││
│  │  │  │ SimpleImputer(median) │  │  │  │ SimpleImputer(most_frequent)│  │││
│  │  │  └───────────────────────┘  │  │  └──────────────────────────────┘  │││
│  │  │            ↓                │  │               ↓                    │││
│  │  │  ┌───────────────────────┐  │  │  ┌──────────────────────────────┐  │││
│  │  │  │   StandardScaler      │  │  │  │   OneHotEncoder              │  │││
│  │  │  └───────────────────────┘  │  │  └──────────────────────────────┘  │││
│  │  └─────────────────────────────┘  └────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING LAYER                                    │
│                                                                              │
│  ┌────────────────────┐    ┌────────────────────┐                           │
│  │ LogisticRegression │    │   RandomForest     │                           │
│  │ (max_iter=1000)    │    │ (n_estimators=200) │                           │
│  └────────────────────┘    └────────────────────┘                           │
│            │                        │                                        │
│            └────────┬───────────────┘                                        │
│                     ▼                                                        │
│           ┌─────────────────┐     ┌─────────────────────────────────────┐   │
│           │ Model Selection │ →   │ MLflow Tracking                     │   │
│           │ (Best ROC-AUC)  │     │ - Parameters, Metrics, Artifacts    │   │
│           └─────────────────┘     └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SERVING LAYER                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                     Full Pipeline (joblib serialized)                    ││
│  │  ┌───────────────────────────┐    ┌─────────────────────────────────┐   ││
│  │  │     Preprocessor          │ →  │    Best Model (Classifier)      │   ││
│  │  │   (ColumnTransformer)     │    │    (LogisticRegression)         │   ││
│  │  └───────────────────────────┘    └─────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         FastAPI Application                              ││
│  │   POST /predict  →  Validate JSON  →  Pipeline.predict()  →  Response   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. MLflow Integration

### What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version and manage models
- **Model Serving**: Deploy models as REST APIs

### How We Use MLflow

#### Configuration (`src/config.py`)
```python
MLFLOW_TRACKING_URI = "mlruns"          # Local directory for tracking
MLFLOW_EXPERIMENT_NAME = "heart-disease-prediction"
```

#### Setup in Training (`src/train.py`)
```python
def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
```

#### What Gets Logged

For each model training run, MLflow logs:

| Category | Items Logged |
|----------|--------------|
| **Parameters** | Model hyperparameters (max_iter, n_estimators, max_depth, etc.) |
| **Metrics** | accuracy, precision, recall, f1_score, roc_auc, cv_roc_auc_mean |
| **Artifacts** | Confusion matrix plots, ROC curves, classification reports, trained model |

#### Training Flow with MLflow

```python
with mlflow.start_run(run_name=f"{model_name}_{timestamp}"):
    # 1. Log parameters
    mlflow.log_params(model.get_params())
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 5)
    
    # 2. Train model
    model.fit(X_train, y_train)
    
    # 3. Evaluate and log metrics
    metrics = evaluate_model(model, X_test, y_test)
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    # 4. Log artifacts (plots)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("roc_curve.png")
    
    # 5. Log the model itself
    mlflow.sklearn.log_model(model, f"model_{model_name}")
```

#### Viewing MLflow UI

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser: http://localhost:5000
```

The UI shows:
- All experiment runs
- Comparison of metrics across runs
- Downloadable artifacts
- Model versioning

---

## 3. Data Pipeline

### Data Acquisition (`src/data_loader.py`)

```
UCI Repository  →  fetch_ucirepo(id=45)  →  pandas DataFrame  →  Local CSV
```

#### Flow:
1. **Check Local Cache**: If `data/heart_disease_raw.csv` exists, load it
2. **Fetch from UCI**: If not cached, download from UCI ML Repository
3. **Save Locally**: Cache the dataset for future runs

```python
def prepare_dataset() -> pd.DataFrame:
    if RAW_DATA_PATH.exists():
        return load_raw_data(RAW_DATA_PATH)  # Use cache
    
    X, y = fetch_dataset_from_uci()  # Download
    save_raw_data(X, y)               # Cache
    return combined_dataframe
```

### Dataset Structure

| Feature | Type | Description |
|---------|------|-------------|
| age | int | Age in years |
| sex | int | 0=female, 1=male |
| cp | int | Chest pain type (0-3) |
| trestbps | int | Resting blood pressure |
| chol | int | Serum cholesterol |
| fbs | int | Fasting blood sugar > 120 |
| restecg | int | Resting ECG results |
| thalach | int | Maximum heart rate |
| exang | int | Exercise induced angina |
| oldpeak | float | ST depression |
| slope | int | Slope of ST segment |
| ca | float | Number of vessels (0-4) |
| thal | float | Thalassemia type |
| **target** | int | 0=no disease, 1-4=disease |

### Target Conversion

Original target has values 0-4. We convert to binary:
```python
df['target'] = (df['target'] > 0).astype(int)
# 0 → 0 (No heart disease)
# 1,2,3,4 → 1 (Heart disease present)
```

---

## 4. Model Training Workflow

### Complete Training Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        python -m src.train                                │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 1: Setup MLflow                                                      │
│   mlflow.set_tracking_uri("mlruns")                                       │
│   mlflow.set_experiment("heart-disease-prediction")                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 2: Load Data                                                         │
│   df = prepare_dataset()  # 303 samples, 13 features                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 3: Preprocess                                                        │
│   - Handle missing values (ca: median, thal: mode)                        │
│   - Convert target to binary                                              │
│   - Split: 80% train (242), 20% test (61)                                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 4: Create & Fit Preprocessor                                         │
│   preprocessor = ColumnTransformer([                                      │
│       ('num', [Imputer, Scaler], numeric_cols),                           │
│       ('cat', [Imputer, OneHotEncoder], categorical_cols)                 │
│   ])                                                                      │
│   X_train_processed = preprocessor.fit_transform(X_train)                 │
│   X_test_processed = preprocessor.transform(X_test)                       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 5: Train Models (with MLflow logging)                                │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │ Model 1: LogisticRegression(max_iter=1000)                          ││
│   │   → Train on X_train_processed                                       ││
│   │   → 5-fold Cross-validation                                          ││
│   │   → Evaluate on X_test_processed                                     ││
│   │   → Log to MLflow                                                    ││
│   └─────────────────────────────────────────────────────────────────────┘│
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │ Model 2: RandomForest(n_estimators=200, max_depth=5)                ││
│   │   → Train, Evaluate, Log                                             ││
│   └─────────────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 6: Select Best Model                                                 │
│   Compare by ROC-AUC score                                                │
│   Winner: LogisticRegression (ROC-AUC: 0.9578)                            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 7: Save Artifacts                                                    │
│   models/preprocessor.pkl      ← Fitted ColumnTransformer                 │
│   models/heart_disease_model.pkl ← Best trained model                     │
│   models/full_pipeline.pkl     ← Complete Pipeline (preprocessor + model) │
└──────────────────────────────────────────────────────────────────────────┘
```

### Model Evaluation Metrics

```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
```

### Results (from actual training)

| Model | Accuracy | ROC-AUC | CV ROC-AUC |
|-------|----------|---------|------------|
| LogisticRegression | 0.8689 | 0.9578 | 0.8995 |
| RandomForest | 0.8852 | 0.9556 | 0.8915 |

**Winner**: LogisticRegression (higher ROC-AUC and CV stability)

---

## 5. Sklearn Pipeline Architecture

### Why Use Pipelines?

1. **Reproducibility**: Same preprocessing for training and inference
2. **No Data Leakage**: Preprocessor fitted only on training data
3. **Single Artifact**: One file for entire prediction flow
4. **Easy Deployment**: Load once, predict many times

### Pipeline Structure

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Numeric features: age, trestbps, chol, thalach, oldpeak, ca
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features: sex, cp, fbs, restecg, exang, slope, thal
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessor
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

# Full pipeline (saved as full_pipeline.pkl)
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
```

### Feature Transformation

```
Input (13 features):
┌─────┬─────┬────┬──────────┬──────┬─────┬─────────┬─────────┬───────┬─────────┬───────┬────┬──────┐
│ age │ sex │ cp │ trestbps │ chol │ fbs │ restecg │ thalach │ exang │ oldpeak │ slope │ ca │ thal │
└─────┴─────┴────┴──────────┴──────┴─────┴─────────┴─────────┴───────┴─────────┴───────┴────┴──────┘
        │
        ▼ ColumnTransformer
        │
┌───────┴───────────────────────────────────────────────────────────────────────────────┐
│                                                                                        │
│  Numeric (6 features) → Impute → Scale → [6 scaled values]                            │
│                                                                                        │
│  Categorical (7 features) → Impute → OneHot → [~16 binary values]                     │
│                                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
Output (22 features):
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ age_scaled, trestbps_scaled, ..., sex_0, sex_1, cp_0, cp_1, cp_2, cp_3, ...        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Inference Flow

```python
# Load once at startup
pipeline = joblib.load("models/full_pipeline.pkl")

# Predict (handles all preprocessing internally)
def predict(patient_data: dict):
    df = pd.DataFrame([patient_data])
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0]
    return {
        "prediction": prediction,
        "confidence": max(probability)
    }
```

---

## 6. API Design

### FastAPI Application (`api/app.py`)

```python
from fastapi import FastAPI
from src.predict import HeartDiseasePredictor

app = FastAPI(title="Heart Disease Prediction API")
predictor = HeartDiseasePredictor().load()

@app.post("/predict")
async def predict(patient_data: PatientData):
    result = predictor.predict(patient_data.model_dump())
    return PredictionResponse(**result)
```

### Request Validation (`api/schemas.py`)

```python
class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    # ... all 13 features with validation
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check + API info |
| `/health` | GET | Health check for orchestration |
| `/predict` | POST | Make prediction |
| `/model/info` | GET | Model metadata |

### Request/Response Example

```bash
# Request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,...}'

# Response
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "probability_no_disease": 0.81,
  "probability_disease": 0.19,
  "confidence": 0.81
}
```

---

## 7. Docker Configuration

### Dockerfile Explained

```dockerfile
# ============================================================
# Stage 1: Builder
# Purpose: Install dependencies in isolated environment
# ============================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build tools (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (Docker layer caching)
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Production
# Purpose: Minimal runtime image
# ============================================================
FROM python:3.11-slim as production

WORKDIR /app

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder (smaller than re-installing)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/    # Pre-trained model included

# Create directories and set permissions
RUN mkdir -p /app/data /app/artifacts /app/mlruns && \
    chown -R appuser:appuser /app

# Switch to non-root user (security best practice)
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1           # Don't buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1    # Don't create .pyc files
ENV PYTHONPATH=/app              # Add /app to Python path

# Expose API port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Multi-Stage Build Benefits

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Builder (~1.2GB)                                        │
│   - Full Python + build tools                                    │
│   - All pip packages installed                                   │
│   - Only used during build                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ COPY only /opt/venv
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Production (~400MB)                                     │
│   - Slim Python base                                             │
│   - Only runtime dependencies                                    │
│   - Application code + trained model                             │
│   - This is what gets deployed                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Docker Compose (`docker-compose.yml`)

```yaml
services:
  # Main API service
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro    # Mount models (read-only)
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s

  # MLflow UI (optional, for experiment tracking)
  mlflow:
    image: python:3.11-slim
    command: mlflow server --host 0.0.0.0 --port 5000
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/artifacts
```

### Docker Commands

```bash
# Build the image
docker build -t heart-disease-predictor:latest .

# Run the container
docker run -d \
  --name heart-disease-api \
  -p 8000:8000 \
  heart-disease-predictor:latest

# Test the container
curl http://localhost:8000/health

# View logs
docker logs heart-disease-api

# Stop and remove
docker stop heart-disease-api
docker rm heart-disease-api
```

---

## 8. CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/ci-cd.yml`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CI/CD Pipeline Flow                                │
└─────────────────────────────────────────────────────────────────────────────┘

  Push/PR to main
        │
        ▼
┌───────────────┐
│    Lint       │  flake8, black, isort
└───────────────┘
        │
        ▼
┌───────────────┐
│    Test       │  pytest with coverage
└───────────────┘
        │
        ▼
┌───────────────┐
│    Train      │  python -m src.train
└───────────────┘
        │
        ▼
┌───────────────┐
│ Docker Build  │  Build + test container
└───────────────┘
        │
        ▼
┌───────────────┐
│ Integration   │  Test /predict endpoint
│    Tests      │
└───────────────┘
        │
        ▼
     SUCCESS ✓
```

### Pipeline Stages

| Stage | Actions | Artifacts |
|-------|---------|-----------|
| **Lint** | flake8, black --check, isort --check | - |
| **Test** | pytest --cov | coverage-report/ |
| **Train** | python -m src.train | models/, mlruns/ |
| **Docker Build** | docker build + test | docker-image.tar.gz |
| **Integration** | curl /health, /predict | - |

---

## Summary

This MLOps pipeline demonstrates:

1. **Reproducible Data Pipeline**: Automated data acquisition and caching
2. **Experiment Tracking**: Full MLflow integration with metrics, parameters, and artifacts
3. **Robust Preprocessing**: Sklearn ColumnTransformer handles all feature engineering
4. **Production-Ready Serving**: FastAPI with validation, logging, and health checks
5. **Container-Ready**: Multi-stage Docker build with security best practices
6. **Automated CI/CD**: GitHub Actions for testing, training, and building

The entire pipeline can be run from scratch with:
```bash
make install    # Install dependencies
make train      # Train model with MLflow
make test       # Run unit tests
make run        # Start API locally
make docker-build  # Build container
```

