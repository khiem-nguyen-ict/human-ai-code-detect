.PHONY: help install data train evaluate deploy test lint clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	pip install -r requirements.txt

data: ## Run data ingestion and validation
	python -m src.data.ingest
	python -m src.data.validate

features: ## Run feature engineering
	python -m src.features.build_features

train: ## Train the model
	python -m src.models.train

evaluate: ## Evaluate the trained model
	python -m src.models.evaluate

registry: ## Register the model
	python -m src.models.registry

deploy: ## Build and start the inference API
	docker build -t human-ai-detect -f docker/Dockerfile .
	docker run -p 8080:8080 --env-file .env human-ai-detect

test: ## Run all tests
	pytest tests/ -v

lint: ## Run linting
	ruff check src/ tests/

clean: ## Remove generated artifacts
	rm -rf models/checkpoints/* models/onnx/* models/registry/mlflow.db
	rm -rf data/processed/*
	rm -rf .mlflow/

retrain: ## Run the full retraining pipeline
	python -m src.retrain.pipeline