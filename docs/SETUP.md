# 🚀 Local Setup Guide

> **For Instructors/Evaluators**: Follow these steps to run the complete MLOps pipeline locally.

---

## Prerequisites

Before starting, ensure you have:

| Tool | Version | Check Command |
|------|---------|---------------|
| Python | 3.9+ (recommended 3.11) | `python --version` |
| pip | Latest | `pip --version` |
| Docker | Latest | `docker --version` |
| kubectl | Latest | `kubectl version --client` |
| Minikube | Latest | `minikube version` |
| Helm | Latest | `helm version` |

### Install Missing Tools (macOS)

```bash
# Install Minikube
brew install minikube

# Install Helm
brew install helm

# Docker Desktop includes kubectl
# Or install separately: brew install kubectl
```

---

## Step 1: Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/deb-sahu/heart-disease-predictor.git
cd heart-disease-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Train the Model

```bash
# Train the model (this also downloads the dataset)
PYTHONPATH=. python scripts/run_training.py
```

**Expected Output:**
```
INFO:src.train:Training LogisticRegression...
INFO:src.train:LogisticRegression - Accuracy: 0.8689, ROC-AUC: 0.9578
...
Best Model: LogisticRegression
```

**Verify model files created:**
```bash
ls -la models/
# Should see: full_pipeline.pkl, heart_disease_model.pkl, preprocessor.pkl
```

---

## Step 3: Run API Locally (Quick Test)

```bash
# Start the API server
PYTHONPATH=. uvicorn api.app:app --host 0.0.0.0 --port 8000
```

**Test the API (in another terminal):**

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Heart Disease",
  "probability_no_disease": 0.81,
  "probability_disease": 0.19,
  "confidence": 0.81
}
```

**Stop the API:** Press `Ctrl+C`

---

## Step 4: View MLflow Experiments

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Open http://localhost:5000 in your browser to view experiment runs.

---

## Step 5: Run Unit Tests

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run with coverage
PYTHONPATH=. pytest tests/ -v --cov=src --cov=api
```

---

## Step 6: Docker Build & Run

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest

# Test the container
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

# View logs
docker logs heart-disease-api

# Stop and remove container
docker stop heart-disease-api && docker rm heart-disease-api
```

---

## Step 7: Kubernetes Deployment (Minikube)

### 7.1 Start Minikube

```bash
# Start Minikube cluster
minikube start

# Use Minikube's Docker daemon
eval $(minikube docker-env)
```

### 7.2 Build & Deploy

```bash
# Build image in Minikube's Docker
docker build -t heart-disease-api:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for pods to be ready
kubectl -n heart-disease-predictor get pods -w
# Press Ctrl+C when pods show "Running"
```

### 7.3 Access the API

```bash
# Port forward to access the API
kubectl -n heart-disease-predictor port-forward svc/heart-disease-api-service 8080:80 &

# Test endpoints
curl http://localhost:8080/health
curl http://localhost:8080/metrics | head -20
```

### 7.4 View Deployment Status

```bash
# All resources
kubectl -n heart-disease-predictor get all

# Pod logs
kubectl -n heart-disease-predictor logs deployment/heart-disease-api
```

---

## Step 8: Monitoring (Prometheus + Grafana)

### 8.1 Deploy Monitoring Stack

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace \
  -f k8s/prometheus-values.yaml

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=false \
  --set adminPassword=admin123

# Wait for pods
kubectl -n monitoring wait --for=condition=ready pod --all --timeout=180s
```

### 8.2 Access Dashboards

```bash
# Port forward Prometheus
kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &

# Port forward Grafana
kubectl -n monitoring port-forward svc/grafana 3000:80 &
```

| Service | URL | Login |
|---------|-----|-------|
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / admin123 |

### 8.3 Configure Grafana

1. Open http://localhost:3000
2. Login: `admin` / `admin123`
3. Go to **Connections** → **Data Sources** → **Add data source**
4. Select **Prometheus**
5. URL: `http://prometheus-server.monitoring.svc.cluster.local`
6. Click **Save & Test**

### 8.4 Verify Metrics

```bash
# Check Prometheus is scraping our API
curl -s "http://localhost:9090/api/v1/targets" | jq '.data.activeTargets[] | select(.labels.job=="heart-disease-api") | {job: .labels.job, health: .health}'

# Query our custom metrics
curl -s "http://localhost:9090/api/v1/query?query=heart_disease_predictions_total" | jq '.data.result'
```

---

## Step 9: Cleanup

```bash
# Stop port forwards
pkill -f "port-forward"

# Delete Kubernetes resources
kubectl delete namespace heart-disease-predictor
kubectl delete namespace monitoring

# Stop Minikube
minikube stop

# (Optional) Delete Minikube cluster
minikube delete
```

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Train model | `PYTHONPATH=. python scripts/run_training.py` |
| Run API locally | `PYTHONPATH=. uvicorn api.app:app --port 8000` |
| Run tests | `PYTHONPATH=. pytest tests/ -v` |
| MLflow UI | `mlflow ui --port 5000` |
| Docker build | `docker build -t heart-disease-api:latest .` |
| K8s deploy | `kubectl apply -f k8s/` |
| K8s status | `kubectl -n heart-disease-predictor get all` |
| K8s logs | `kubectl -n heart-disease-predictor logs deployment/heart-disease-api` |
| Port forward API | `kubectl -n heart-disease-predictor port-forward svc/heart-disease-api-service 8080:80` |

---

## Troubleshooting

### Model not found error
```bash
# Ensure you've trained the model first
PYTHONPATH=. python scripts/run_training.py
```

### Docker build fails
```bash
# Make sure Docker is running
docker info
```

### Minikube issues
```bash
# Reset Minikube
minikube delete
minikube start
```

### Port already in use
```bash
# Find and kill process using the port
lsof -i :8000
kill -9 <PID>
```

