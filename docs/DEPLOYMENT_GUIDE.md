# Deployment & Monitoring Guide

This guide covers deploying the Heart Disease Predictor API to local Kubernetes and setting up free monitoring with Grafana Cloud.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Local Kubernetes Deployment](#2-local-kubernetes-deployment)
3. [Grafana Cloud Setup (Free)](#3-grafana-cloud-setup-free)
4. [Prometheus Metrics](#4-prometheus-metrics)
5. [Creating Dashboards](#5-creating-dashboards)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### Required Software

| Tool | Purpose | Installation |
|------|---------|--------------|
| Docker | Container runtime | [Install Docker](https://docs.docker.com/get-docker/) |
| kubectl | Kubernetes CLI | [Install kubectl](https://kubernetes.io/docs/tasks/tools/) |
| Minikube OR Docker Desktop | Local Kubernetes | See below |

### Option A: Minikube (Recommended for Learning)

```bash
# macOS
brew install minikube

# Linux
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Windows (PowerShell as Admin)
choco install minikube
```

### Option B: Docker Desktop Kubernetes

1. Open Docker Desktop
2. Go to Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"

---

## 2. Local Kubernetes Deployment

### Quick Deploy (Automated)

```bash
# Make the script executable
chmod +x scripts/deploy_k8s.sh

# Deploy to Minikube
./scripts/deploy_k8s.sh minikube

# OR deploy to Docker Desktop
./scripts/deploy_k8s.sh docker-desktop
```

### Manual Deployment Steps

#### Step 1: Start Kubernetes Cluster

**Minikube:**
```bash
# Start with adequate resources
minikube start --driver=docker --memory=4096 --cpus=2

# Enable ingress controller
minikube addons enable ingress

# Use Minikube's Docker daemon (so images are accessible)
eval $(minikube docker-env)
```

**Docker Desktop:**
```bash
# Verify Kubernetes is running
kubectl cluster-info
```

#### Step 2: Build Docker Image

```bash
# Build the image (use minikube's Docker daemon)
docker build -t heart-disease-api:latest .

# Verify image was created
docker images | grep heart-disease-api
```

> **Important**: Make sure to run `eval $(minikube docker-env)` before building so the image is available in minikube's Docker registry.

#### Step 3: Deploy to Kubernetes

```bash
# Apply all Kubernetes manifests
kubectl apply -k k8s/

# OR apply individually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

#### Step 4: Verify Deployment

```bash
# Check pods are running
kubectl get pods -n heart-disease-predictor

# Expected output:
# NAME                                 READY   STATUS    RESTARTS   AGE
# heart-disease-api-xxxxxxxxxx-xxxxx   1/1     Running   0          1m
# heart-disease-api-xxxxxxxxxx-xxxxx   1/1     Running   0          1m

# Check services
kubectl get svc -n heart-disease-predictor

# Check deployment status
kubectl describe deployment heart-disease-api -n heart-disease-predictor
```

#### Step 5: Access the API

**Minikube:**
```bash
# Get service URL
minikube service heart-disease-api-service -n heart-disease-predictor --url

# OR use port-forward
kubectl port-forward svc/heart-disease-api-service 8080:80 -n heart-disease-predictor

# Access at http://localhost:8080
```

**Docker Desktop:**
```bash
# Use port-forward
kubectl port-forward svc/heart-disease-api-service 8080:80 -n heart-disease-predictor

# Access at http://localhost:8080
```

#### Step 6: Test the API

**Available Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns model status |
| `/predict` | POST | Make heart disease prediction |
| `/metrics` | GET | Prometheus metrics for monitoring |
| `/docs` | GET | Interactive Swagger UI documentation |
| `/model/info` | GET | Model information and features |

**Test Commands:**

```bash
# 1. Health check
curl http://localhost:8080/health
# Response: {"status":"healthy","model_loaded":true,"version":"1.0.0"}

# 2. Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": 1
  }'
# Response: {"prediction":0,"prediction_label":"No Heart Disease","probability_no_disease":0.81,"probability_disease":0.19,"confidence":0.81}

# 3. Check Prometheus metrics
curl http://localhost:8080/metrics | grep heart_disease
# Shows custom metrics like heart_disease_predictions_total, heart_disease_prediction_latency_seconds

# 4. Get model info
curl http://localhost:8080/model/info

# 5. Open Swagger UI in browser
open http://localhost:8080/docs
```

**Prediction Input Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `age` | int | Age in years (29-77) |
| `sex` | int | Sex (0=female, 1=male) |
| `cp` | int | Chest pain type (0-3) |
| `trestbps` | int | Resting blood pressure (mm Hg) |
| `chol` | int | Serum cholesterol (mg/dl) |
| `fbs` | int | Fasting blood sugar > 120 mg/dl (0/1) |
| `restecg` | int | Resting ECG results (0-2) |
| `thalach` | int | Max heart rate achieved |
| `exang` | int | Exercise induced angina (0/1) |
| `oldpeak` | float | ST depression induced by exercise |
| `slope` | int | Slope of peak exercise ST segment (0-2) |
| `ca` | int | Number of major vessels (0-4) |
| `thal` | int | Thalassemia (1=normal, 2=fixed defect, 3=reversible) |

---

## 3. Local Prometheus & Grafana Setup

We deploy Prometheus and Grafana locally in Minikube using Helm charts. This provides a **completely free** monitoring solution without any cloud accounts.

### Prerequisites

- Minikube running (from Step 2)
- Helm installed (`brew install helm`)

### Step 1: Add Helm Repositories

```bash
# Add Prometheus and Grafana Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

### Step 2: Deploy Prometheus

We use a custom values file (`k8s/prometheus-values.yaml`) that configures Prometheus to scrape our heart-disease-api metrics:

```bash
# Install Prometheus with custom config
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace \
  -f k8s/prometheus-values.yaml
```

The `prometheus-values.yaml` includes:
- Auto-discovery of our API pods
- Scraping `/metrics` endpoint every 15 seconds
- Kubernetes node and pod metrics

### Step 3: Deploy Grafana

```bash
# Install Grafana with a known password
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set persistence.enabled=false \
  --set adminPassword=admin123
```

### Step 4: Wait for Pods to be Ready

```bash
# Wait for all monitoring pods
kubectl -n monitoring wait --for=condition=ready pod --all --timeout=120s

# Verify pods are running
kubectl -n monitoring get pods
```

Expected output:
```
NAME                                            READY   STATUS    RESTARTS   AGE
grafana-xxxxxxxxxx-xxxxx                        1/1     Running   0          1m
prometheus-server-xxxxxxxxxx-xxxxx              2/2     Running   0          1m
prometheus-kube-state-metrics-xxxxxxxxxx-xxxxx  1/1     Running   0          1m
prometheus-prometheus-node-exporter-xxxxx       1/1     Running   0          1m
```

### Step 5: Access the Services

```bash
# Port forward Prometheus (in background)
kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &

# Port forward Grafana (in background)
kubectl -n monitoring port-forward svc/grafana 3000:80 &
```

**Access URLs:**

| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | `admin` / `admin123` |

### Step 6: Configure Grafana Data Source

1. Open http://localhost:3000
2. Login with `admin` / `admin123`
3. Go to **Connections** → **Data Sources** → **Add data source**
4. Select **Prometheus**
5. Set URL to: `http://prometheus-server.monitoring.svc.cluster.local`
6. Click **Save & Test** (should show "Data source is working")

### Step 7: Verify Metrics are Being Collected

```bash
# Check Prometheus targets
curl -s "http://localhost:9090/api/v1/targets" | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Query our custom metrics
curl -s "http://localhost:9090/api/v1/query?query=heart_disease_predictions_total" | jq '.data.result'
```

### Quick Setup Script

For convenience, here's a complete setup script:

```bash
#!/bin/bash
# setup-monitoring.sh

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
kubectl -n monitoring wait --for=condition=ready pod --all --timeout=120s

echo "Monitoring stack deployed!"
echo "Prometheus: kubectl -n monitoring port-forward svc/prometheus-server 9090:80"
echo "Grafana:    kubectl -n monitoring port-forward svc/grafana 3000:80"
```

---

## 4. Prometheus Metrics

The API exposes the following custom metrics at `/metrics`:

### Prediction Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `heart_disease_predictions_total` | Counter | Total predictions by result |
| `heart_disease_prediction_latency_seconds` | Histogram | Prediction request latency |
| `heart_disease_prediction_confidence` | Histogram | Prediction confidence distribution |
| `heart_disease_model_loaded` | Gauge | Whether model is loaded (1/0) |
| `heart_disease_prediction_errors_total` | Counter | Total errors by type |

### HTTP Metrics (Auto-instrumented)

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | Request duration |
| `heart_disease_inprogress_requests` | Gauge | In-progress requests |

### Example Queries

```promql
# Prediction rate (per minute)
rate(heart_disease_predictions_total[1m])

# Average prediction latency
rate(heart_disease_prediction_latency_seconds_sum[5m]) 
/ rate(heart_disease_prediction_latency_seconds_count[5m])

# Error rate
rate(heart_disease_prediction_errors_total[5m])

# Model uptime
heart_disease_model_loaded

# Request rate by status code
rate(http_requests_total[1m])
```

---

## 5. Creating Grafana Dashboards

### Import Pre-built Dashboard

1. In Grafana, go to **Dashboards** → **Import**
2. Paste the JSON from `grafana-dashboard.json` below
3. Select your Prometheus data source
4. Click **Import**

### Create Custom Dashboard

1. Go to **Dashboards** → **New Dashboard**
2. Add panels with these visualizations:

#### Panel 1: Predictions Over Time
```promql
sum(rate(heart_disease_predictions_total[5m])) by (prediction_result)
```

#### Panel 2: Average Latency
```promql
histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m]))
```

#### Panel 3: Error Rate
```promql
sum(rate(heart_disease_prediction_errors_total[5m])) by (error_type)
```

#### Panel 4: Model Status
```promql
heart_disease_model_loaded
```

#### Panel 5: Confidence Distribution
```promql
histogram_quantile(0.5, rate(heart_disease_prediction_confidence_bucket[5m]))
```

### Sample Dashboard JSON

Save this as `grafana-dashboard.json`:

```json
{
  "title": "Heart Disease Predictor",
  "panels": [
    {
      "title": "Predictions per Minute",
      "type": "timeseries",
      "targets": [
        {
          "expr": "sum(rate(heart_disease_predictions_total[1m])) by (prediction_result)",
          "legendFormat": "{{prediction_result}}"
        }
      ]
    },
    {
      "title": "P95 Latency",
      "type": "gauge",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(heart_disease_prediction_latency_seconds_bucket[5m]))"
        }
      ]
    },
    {
      "title": "Model Status",
      "type": "stat",
      "targets": [
        {
          "expr": "heart_disease_model_loaded"
        }
      ]
    }
  ]
}
```

---

## 6. Troubleshooting

### Common Issues

#### Pod not starting
```bash
# Check pod status
kubectl describe pod -l app=heart-disease-api -n heart-disease-predictor

# Check logs
kubectl logs -l app=heart-disease-api -n heart-disease-predictor
```

#### Image not found
```bash
# For Minikube, ensure you're using Minikube's Docker
eval $(minikube docker-env)
docker build -t heart-disease-predictor:latest .

# Set imagePullPolicy to IfNotPresent in deployment.yaml
```

#### Service not accessible
```bash
# Check service endpoints
kubectl get endpoints -n heart-disease-predictor

# Use port-forward as fallback
kubectl port-forward svc/heart-disease-api-service 8080:80 -n heart-disease-predictor
```

#### Metrics not appearing in Grafana
```bash
# Verify metrics endpoint works
kubectl port-forward svc/heart-disease-api-internal 8000:8000 -n heart-disease-predictor
curl http://localhost:8000/metrics

# Check Prometheus agent logs
kubectl logs -l app=prometheus-agent -n heart-disease-predictor
```

### Cleanup

```bash
# Delete the entire namespace (removes all resources)
kubectl delete namespace heart-disease-predictor

# Stop Minikube
minikube stop

# Delete Minikube cluster
minikube delete
```

---

## Quick Reference

### Deployment Commands

```bash
# Deploy
kubectl apply -k k8s/

# Check status
kubectl get all -n heart-disease-predictor

# View logs
kubectl logs -f deployment/heart-disease-api -n heart-disease-predictor

# Scale
kubectl scale deployment heart-disease-api --replicas=3 -n heart-disease-predictor

# Update image
kubectl set image deployment/heart-disease-api heart-disease-api=heart-disease-predictor:v2 -n heart-disease-predictor

# Delete
kubectl delete namespace heart-disease-predictor
```

### URLs

| Service | URL |
|---------|-----|
| API (via port-forward) | http://localhost:8080 |
| Health Check | http://localhost:8080/health |
| API Docs | http://localhost:8080/docs |
| Metrics | http://localhost:8080/metrics |
| Grafana Cloud | https://your-stack.grafana.net |

