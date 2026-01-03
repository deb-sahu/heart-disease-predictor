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
# Build the image
docker build -t heart-disease-predictor:latest .

# Verify image was created
docker images | grep heart-disease
```

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

```bash
# Health check
curl http://localhost:8080/health

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'

# Check metrics endpoint
curl http://localhost:8080/metrics
```

---

## 3. Grafana Cloud Setup (Free)

Grafana Cloud offers a **generous free tier** that includes:
- 10,000 series for Prometheus metrics
- 50 GB logs
- 50 GB traces
- 14-day retention
- 3 users

### Step 1: Create Free Account

1. Go to [grafana.com/auth/sign-up/create-user](https://grafana.com/auth/sign-up/create-user)
2. Sign up with email or GitHub
3. Choose the **Free** plan

### Step 2: Set Up Prometheus Integration

1. After logging in, you'll see the **Getting Started Guide**
2. Go to **Connections** → **Add new connection** in the left sidebar
3. Search for "Prometheus" and click on it
4. Or navigate to the **Prometheus onboarding** page

On the Prometheus onboarding page, you'll see a wizard with multiple steps:

**Step 1 - "How do you want to get started?"**
- Select **"Collect and send metrics to a fully-managed Prometheus Stack"**
- Click **Next**

**Step 2 - "Which services generate your data?"**
- You'll see options like AWS CloudWatch, Azure, Docker, etc.
- Scroll down and select **"Custom Setup Options"** (this is for our custom FastAPI app metrics)
- Click **Next**

### Step 3: Get Your Credentials

After selecting **Custom Setup Options**, you'll see the configuration page with your credentials.

The page displays:

1. **Prometheus Remote Write Endpoint** - The URL where metrics will be sent
   - Example: `https://prometheus-prod-24-prod-eu-west-2.grafana.net/api/prom/push`
   - Copy this value

2. **Username** - Your numeric Grafana Cloud username
   - Example: `123456`
   - Copy this value

3. **Password / API Token** - Click **"Generate now"** button to create a new API token
   - Give it a descriptive name like `heart-disease-api-metrics`
   - Copy and save the token immediately (it won't be shown again!)

> **Important**: Save these three values securely. You'll need them to configure Prometheus to send metrics to Grafana Cloud.

**Alternative Method** (if you need to find credentials later):
1. Go to **Home** → Click on your stack name (e.g., "yourname")
2. Under **Prometheus**, click **Details** or **Send Metrics**
3. Copy the **Remote Write Endpoint** and **Username**
4. Generate an API key with `MetricsPublisher` role from **Security** → **API Keys**

### Step 4: Configure Prometheus Agent

Create a Prometheus configuration to send metrics to Grafana Cloud:

```yaml
# prometheus-agent.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-agent-config
  namespace: heart-disease-predictor
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s

    scrape_configs:
      - job_name: 'heart-disease-api'
        static_configs:
          - targets: ['heart-disease-api-internal:8000']
        metrics_path: /metrics

    remote_write:
      - url: YOUR_GRAFANA_CLOUD_PROMETHEUS_URL
        basic_auth:
          username: YOUR_USERNAME
          password: YOUR_API_KEY
```

### Step 5: Deploy Prometheus Agent

```bash
# Create the ConfigMap (after editing with your credentials)
kubectl apply -f prometheus-agent.yaml

# Deploy Prometheus agent
kubectl apply -f k8s/prometheus-agent.yaml
```

### Alternative: Use Grafana Alloy (Simpler)

Grafana recommends using **Alloy** (formerly Grafana Agent) for collecting metrics:

```bash
# Install Alloy using Helm
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install alloy grafana/alloy \
  --namespace heart-disease-predictor \
  --set alloy.config="
    prometheus.scrape \"api\" {
      targets = [{\"__address__\" = \"heart-disease-api-internal:8000\"}]
      forward_to = [prometheus.remote_write.grafana_cloud.receiver]
      metrics_path = \"/metrics\"
    }
    
    prometheus.remote_write \"grafana_cloud\" {
      endpoint {
        url = \"YOUR_GRAFANA_CLOUD_URL\"
        basic_auth {
          username = \"YOUR_USERNAME\"
          password = \"YOUR_API_KEY\"
        }
      }
    }
  "
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

## 5. Creating Dashboards

### Import Pre-built Dashboard

1. In Grafana Cloud, go to **Dashboards** → **Import**
2. Use dashboard ID or paste JSON

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

