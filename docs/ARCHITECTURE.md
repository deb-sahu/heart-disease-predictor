# System Architecture

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           HEART DISEASE PREDICTOR - MLOps                        │
│                              System Architecture                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 DEVELOPMENT                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Source    │    │    Unit     │    │   Model     │    │   MLflow    │        │
│  │    Code     │───▶│   Tests     │───▶│  Training   │───▶│  Tracking   │        │
│  │  (Python)   │    │  (Pytest)   │    │             │    │             │        │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘        │
│        │                                      │                  │               │
│        │                                      ▼                  ▼               │
│        │                              ┌─────────────┐    ┌─────────────┐         │
│        │                              │   Model     │    │  Metrics &  │         │
│        │                              │ Artifacts   │    │  Parameters │         │
│        │                              │  (.pkl)     │    │             │         │
│        │                              └─────────────┘    └─────────────┘         │
│        ▼                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │                        GitHub Repository                             │         │
│  └─────────────────────────────────────────────────────────────────────┘         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              CI/CD PIPELINE (GitHub Actions)                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│  │  Lint   │───▶│  Test   │───▶│  Train  │───▶│  Build  │───▶│  Push   │         │
│  │ (flake8)│    │(pytest) │    │ (Model) │    │(Docker) │    │ (Image) │         │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           CONTAINER REGISTRY                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│                    ┌──────────────────────────────────────┐                      │
│                    │     heart-disease-api:latest         │                      │
│                    │     (Docker Image)                   │                      │
│                    └──────────────────────────────────────┘                      │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       KUBERNETES CLUSTER (Minikube)                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │                    Namespace: heart-disease-predictor                │         │
│  ├─────────────────────────────────────────────────────────────────────┤         │
│  │                                                                      │         │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │         │
│  │  │    Pod 1     │  │    Pod 2     │  │   Service    │               │         │
│  │  │              │  │              │  │              │               │         │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │  │ LoadBalancer │               │         │
│  │  │ │  FastAPI │ │  │ │  FastAPI │ │  │   :80/8080   │◀──── Users    │         │
│  │  │ │   :8000  │ │  │ │   :8000  │ │  │              │               │         │
│  │  │ └──────────┘ │  │ └──────────┘ │  └──────────────┘               │         │
│  │  │              │  │              │         │                        │         │
│  │  │ ┌──────────┐ │  │ ┌──────────┐ │         │                        │         │
│  │  │ │  Model   │ │  │ │  Model   │ │         │                        │         │
│  │  │ │  (.pkl)  │ │  │ │  (.pkl)  │ │         │                        │         │
│  │  │ └──────────┘ │  │ └──────────┘ │         │                        │         │
│  │  └──────────────┘  └──────────────┘         │                        │         │
│  │         │                  │                │                        │         │
│  │         └────────┬─────────┘                │                        │         │
│  │                  ▼                          │                        │         │
│  │         ┌──────────────┐                    │                        │         │
│  │         │   /metrics   │────────────────────┼──────────┐             │         │
│  │         │   endpoint   │                    │          │             │         │
│  │         └──────────────┘                    │          │             │         │
│  └─────────────────────────────────────────────┼──────────┼─────────────┘         │
│                                                │          │                       │
│  ┌─────────────────────────────────────────────┼──────────┼─────────────┐         │
│  │                    Namespace: monitoring    │          │             │         │
│  ├─────────────────────────────────────────────┼──────────┼─────────────┤         │
│  │                                             │          │             │         │
│  │  ┌──────────────┐                           │          │             │         │
│  │  │  Prometheus  │◀──────────────────────────┘          │             │         │
│  │  │   Server     │  (scrapes /metrics)                  │             │         │
│  │  │   :9090      │                                      │             │         │
│  │  └──────────────┘                                      │             │         │
│  │         │                                              │             │         │
│  │         ▼                                              │             │         │
│  │  ┌──────────────┐                                      │             │         │
│  │  │   Grafana    │◀─────────────────────────────────────┘             │         │
│  │  │  Dashboard   │  (visualizes metrics)                              │         │
│  │  │   :3000      │                                                    │         │
│  │  └──────────────┘                                                    │         │
│  │                                                                      │         │
│  └──────────────────────────────────────────────────────────────────────┘         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘

                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              USER ACCESS                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │    API      │    │   Swagger   │    │ Prometheus  │    │   Grafana   │        │
│  │  :8080      │    │    /docs    │    │   :9090     │    │   :3000     │        │
│  │             │    │             │    │             │    │             │        │
│  │  /health    │    │ Interactive │    │   Metrics   │    │ Dashboards  │        │
│  │  /predict   │    │    Docs     │    │   Query     │    │   Alerts    │        │
│  │  /metrics   │    │             │    │             │    │             │        │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Source Code Layer
- **Python Modules**: `src/` - Core ML code (data loading, preprocessing, training, prediction)
- **API Layer**: `api/` - FastAPI application with REST endpoints
- **Tests**: `tests/` - Unit and integration tests using Pytest

### 2. ML Pipeline
```
Data Acquisition → Preprocessing → Feature Engineering → Model Training → Evaluation → Artifact Storage
      │                │                  │                   │              │              │
      ▼                ▼                  ▼                   ▼              ▼              ▼
   UCI Repo      Handle Missing     StandardScaler      LogisticReg    ROC-AUC        MLflow
                    Values          OneHotEncoder       RandomForest   Accuracy       models/
```

### 3. CI/CD Pipeline (GitHub Actions)

| Stage | Tool | Purpose |
|-------|------|---------|
| Lint | flake8, black, isort | Code quality |
| Test | pytest | Unit tests |
| Train | MLflow | Model training |
| Build | Docker | Container image |
| Push | Registry | Image storage |

### 4. Kubernetes Resources

| Resource | Purpose |
|----------|---------|
| Namespace | Isolation (`heart-disease-predictor`) |
| Deployment | Pod management (2 replicas) |
| Service | Load balancing |
| Ingress | External access |
| HPA | Auto-scaling |
| ConfigMap | Configuration |

### 5. Monitoring Stack

| Component | Port | Purpose |
|-----------|------|---------|
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Visualization |
| API /metrics | 8000 | Metrics endpoint |

## Data Flow

```
1. User Request
      │
      ▼
2. Kubernetes Ingress/Service
      │
      ▼
3. Load Balancer (distributes to pods)
      │
      ▼
4. FastAPI Application
      │
      ├──▶ /health (health check)
      ├──▶ /predict (prediction)
      │         │
      │         ▼
      │    Model Pipeline
      │         │
      │         ├── Preprocessor (StandardScaler, OneHotEncoder)
      │         │
      │         └── Classifier (LogisticRegression)
      │                  │
      │                  ▼
      │         Prediction Result
      │                  │
      │                  ▼
      │         JSON Response
      │
      └──▶ /metrics (Prometheus)
                 │
                 ▼
           Prometheus Server
                 │
                 ▼
           Grafana Dashboard
```

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| **ML** | scikit-learn, pandas, numpy, MLflow |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Monitoring** | prometheus-client, prometheus-fastapi-instrumentator |
| **Container** | Docker, multi-stage builds |
| **Orchestration** | Kubernetes, Helm |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest, coverage |
| **Linting** | flake8, black, isort |

