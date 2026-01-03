#!/bin/bash
# ============================================================================
# Heart Disease Predictor - Kubernetes Deployment Script
# ============================================================================
# This script deploys the Heart Disease Prediction API to a local Kubernetes
# cluster (Minikube or Docker Desktop Kubernetes).
#
# Prerequisites:
#   - Docker installed and running
#   - kubectl installed
#   - Minikube OR Docker Desktop with Kubernetes enabled
#
# Usage:
#   ./scripts/deploy_k8s.sh [minikube|docker-desktop]
# ============================================================================

set -e

CLUSTER_TYPE="${1:-minikube}"
NAMESPACE="heart-disease-predictor"
IMAGE_NAME="heart-disease-predictor"
IMAGE_TAG="latest"

echo "=============================================="
echo "Heart Disease Predictor - K8s Deployment"
echo "=============================================="
echo "Cluster type: $CLUSTER_TYPE"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists kubectl; then
    echo "ERROR: kubectl is not installed. Please install kubectl first."
    exit 1
fi

if ! command_exists docker; then
    echo "ERROR: docker is not installed. Please install Docker first."
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Setup based on cluster type
if [ "$CLUSTER_TYPE" == "minikube" ]; then
    echo "Setting up Minikube..."
    
    if ! command_exists minikube; then
        echo "ERROR: minikube is not installed. Please install minikube first."
        exit 1
    fi
    
    # Check if minikube is running
    if ! minikube status | grep -q "Running"; then
        echo "Starting Minikube..."
        minikube start --driver=docker --memory=4096 --cpus=2
    fi
    
    # Use minikube's Docker daemon
    echo "Configuring Docker to use Minikube's daemon..."
    eval $(minikube docker-env)
    
    # Enable ingress addon
    echo "Enabling ingress addon..."
    minikube addons enable ingress
    
elif [ "$CLUSTER_TYPE" == "docker-desktop" ]; then
    echo "Using Docker Desktop Kubernetes..."
    
    # Verify Kubernetes is running
    if ! kubectl cluster-info >/dev/null 2>&1; then
        echo "ERROR: Kubernetes is not running. Please enable Kubernetes in Docker Desktop."
        exit 1
    fi
    
else
    echo "ERROR: Unknown cluster type. Use 'minikube' or 'docker-desktop'"
    exit 1
fi

echo "✓ Cluster setup complete"
echo ""

# Build Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
echo "✓ Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Deploy to Kubernetes
echo "Deploying to Kubernetes..."

# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply all resources using kustomize
kubectl apply -k k8s/

echo "✓ Kubernetes resources applied"
echo ""

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/heart-disease-api -n $NAMESPACE

echo "✓ Deployment is ready"
echo ""

# Get service information
echo "=============================================="
echo "Deployment Complete!"
echo "=============================================="
echo ""

if [ "$CLUSTER_TYPE" == "minikube" ]; then
    # Get Minikube IP
    MINIKUBE_IP=$(minikube ip)
    SERVICE_URL=$(minikube service heart-disease-api-service -n $NAMESPACE --url 2>/dev/null || echo "")
    
    echo "Access the API:"
    echo "  - Using Minikube service: minikube service heart-disease-api-service -n $NAMESPACE"
    echo "  - Direct URL: $SERVICE_URL"
    echo ""
    echo "For Ingress access, add this to /etc/hosts:"
    echo "  $MINIKUBE_IP heart-disease.local"
    echo ""
    echo "Then access: http://heart-disease.local"
    
else
    # Docker Desktop uses localhost
    echo "Access the API:"
    echo "  - URL: http://localhost (LoadBalancer will forward to the service)"
    echo ""
    echo "If LoadBalancer is pending, use port-forward:"
    echo "  kubectl port-forward svc/heart-disease-api-service 8080:80 -n $NAMESPACE"
    echo "  Then access: http://localhost:8080"
fi

echo ""
echo "Useful commands:"
echo "  - View pods:       kubectl get pods -n $NAMESPACE"
echo "  - View logs:       kubectl logs -f deployment/heart-disease-api -n $NAMESPACE"
echo "  - View services:   kubectl get svc -n $NAMESPACE"
echo "  - Describe deploy: kubectl describe deployment heart-disease-api -n $NAMESPACE"
echo "  - Delete all:      kubectl delete namespace $NAMESPACE"
echo ""
echo "Metrics endpoint: http://<api-url>/metrics"

