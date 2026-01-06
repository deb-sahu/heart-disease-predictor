#!/bin/bash
# setup-monitoring.sh
# Deploys Prometheus and Grafana to Minikube for monitoring the Heart Disease API

set -e

echo "=============================================="
echo "  📊 Setting up Monitoring Stack"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo "❌ Helm is not installed. Please install it first:"
    echo "   brew install helm"
    exit 1
fi

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ kubectl is not configured or cluster is not running."
    echo "   Please start minikube: minikube start"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 1: Adding Helm repositories...${NC}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
helm repo update

echo ""
echo -e "${YELLOW}Step 2: Installing Prometheus...${NC}"
if helm status prometheus -n monitoring &> /dev/null; then
    echo "Prometheus already installed, upgrading..."
    helm upgrade prometheus prometheus-community/prometheus \
        --namespace monitoring \
        -f k8s/prometheus-values.yaml
else
    helm install prometheus prometheus-community/prometheus \
        --namespace monitoring \
        --create-namespace \
        -f k8s/prometheus-values.yaml
fi

echo ""
echo -e "${YELLOW}Step 3: Installing Grafana...${NC}"
if helm status grafana -n monitoring &> /dev/null; then
    echo "Grafana already installed, upgrading..."
    helm upgrade grafana grafana/grafana \
        --namespace monitoring \
        --set persistence.enabled=false \
        --set adminPassword=admin123
else
    helm install grafana grafana/grafana \
        --namespace monitoring \
        --set persistence.enabled=false \
        --set adminPassword=admin123
fi

echo ""
echo -e "${YELLOW}Step 4: Waiting for pods to be ready...${NC}"
kubectl -n monitoring wait --for=condition=ready pod --all --timeout=180s

echo ""
echo -e "${GREEN}=============================================="
echo "  ✅ Monitoring Stack Deployed Successfully!"
echo "==============================================${NC}"
echo ""
echo "📊 Services:"
echo ""
echo "┌─────────────┬───────────────────────┬────────────────────┐"
echo "│ Service     │ URL                   │ Credentials        │"
echo "├─────────────┼───────────────────────┼────────────────────┤"
echo "│ Prometheus  │ http://localhost:9090 │ -                  │"
echo "├─────────────┼───────────────────────┼────────────────────┤"
echo "│ Grafana     │ http://localhost:3000 │ admin / admin123   │"
echo "└─────────────┴───────────────────────┴────────────────────┘"
echo ""
echo "🔗 To access the services, run:"
echo ""
echo "   # Prometheus"
echo "   kubectl -n monitoring port-forward svc/prometheus-server 9090:80 &"
echo ""
echo "   # Grafana"
echo "   kubectl -n monitoring port-forward svc/grafana 3000:80 &"
echo ""
echo "📖 Next Steps:"
echo "   1. Open Grafana at http://localhost:3000"
echo "   2. Login with admin / admin123"
echo "   3. Add Prometheus data source: http://prometheus-server.monitoring.svc.cluster.local"
echo "   4. Import or create dashboards"
echo ""

