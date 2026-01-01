# Heart Disease Predictor - Makefile
# Convenient commands for development and deployment

.PHONY: help install install-dev clean lint format test train run docker-build docker-run docker-stop all

# Default target
help:
	@echo "Heart Disease Predictor - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install all dependencies (including dev)"
	@echo ""
	@echo "Development:"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black and isort"
	@echo "  make test          Run unit tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo ""
	@echo "ML Pipeline:"
	@echo "  make download      Download the dataset"
	@echo "  make train         Train the model"
	@echo "  make mlflow-ui     Start MLflow UI"
	@echo ""
	@echo "API:"
	@echo "  make run           Run the API server locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo "  make docker-stop   Stop Docker container"
	@echo "  make docker-test   Test Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove cache and temporary files"
	@echo "  make clean-all     Remove all generated files"

# ============================================================================
# Setup
# ============================================================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install black isort mypy

# ============================================================================
# Development
# ============================================================================

lint:
	flake8 src/ api/ tests/ --max-line-length=120 --ignore=E501,W503,E203
	@echo "✓ Linting passed!"

format:
	black src/ api/ tests/ --line-length=120
	isort src/ api/ tests/
	@echo "✓ Code formatted!"

test:
	PYTHONPATH=. pytest tests/ -v

test-cov:
	PYTHONPATH=. pytest tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# ============================================================================
# ML Pipeline
# ============================================================================

download:
	PYTHONPATH=. python scripts/download_data.py

train:
	PYTHONPATH=. python -m src.train

mlflow-ui:
	mlflow ui --port 5000

# ============================================================================
# API
# ============================================================================

run:
	PYTHONPATH=. uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker build -t heart-disease-predictor:latest .

docker-run:
	docker run -d --name heart-disease-api -p 8000:8000 heart-disease-predictor:latest
	@echo "API running at http://localhost:8000"
	@echo "Health check: http://localhost:8000/health"
	@echo "API docs: http://localhost:8000/docs"

docker-stop:
	docker stop heart-disease-api || true
	docker rm heart-disease-api || true

docker-test:
	@echo "Testing Docker container..."
	curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@echo "Testing prediction endpoint..."
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}' \
		| python -m json.tool

docker-compose-up:
	docker-compose up -d
	@echo "Services started. API: http://localhost:8000, MLflow: http://localhost:5000"

docker-compose-down:
	docker-compose down

# ============================================================================
# Cleanup
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✓ Cleaned cache and temporary files"

clean-all: clean
	rm -rf models/*.pkl
	rm -rf artifacts/
	rm -rf mlruns/
	rm -rf data/*.csv
	rm -rf logs/
	@echo "✓ Cleaned all generated files"

# ============================================================================
# Full Pipeline
# ============================================================================

all: install download train test docker-build
	@echo "✓ Full pipeline completed!"

