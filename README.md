# Human vs AI Code Detection

Detect whether C source code is human-written or AI-generated using GraphCodeBERT fine-tuned with a linear classification head, exported to ONNX FP16 for offline inference.

## What it does

- Loads a labeled dataset of human-written and AI-generated C code.
- Fine-tunes `microsoft/graphcodebert-base` by freezing the encoder and training a 768 → 2 linear classifier.
- Serializes the fine-tuned model to PyTorch (`graphcodebert_human_ai.pt`).
- Exports to ONNX FP16 (`graphcodebert_human_ai_fp16.onnx`) for CPU-friendly inference.
- Runs predictions on raw C files or code strings, returning Human / AI probabilities.

## Prerequisites

- Python 3.9+
- pip
- (Optional) Docker and Docker Compose for containerized deployment

### Python dependencies

```bash
pip install -r requirements.txt
```

## Dataset

`dataset.zip` should contain:

```
dataset.zip
└── human/   # C files labeled as human-written (label 0)
    └── *.c
└── ai/      # C files labeled as AI-generated  (label 1)
    └── *.c
```

## Quick Start

### Option 1: Makefile pipeline (recommended)

```bash
# 1. Copy environment variables
cp .env.example .env

# 2. Install dependencies
make install

# 3. Run the full pipeline
make data          # Ingest and validate dataset
make features      # Tokenize C files
make train         # Train model and export ONNX
make evaluate      # Compute metrics
make registry      # Register in MLflow

# 4. Run tests
make test

# 5. Start the inference API
make deploy
```

### Option 2: Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Services:
- `train` — runs the full training pipeline
- `serve` — starts the FastAPI inference server on port 8080
- `monitor` — runs drift detection
- `retrain` — orchestrates retraining

### Testing the Docker Setup

To verify the Docker services work end-to-end, follow these steps:

1. **Ensure Docker (Colima) is running**
   ```bash
   brew services start colima
   docker context use colima
   docker info
   ```

2. **Copy environment variables**
   ```bash
   cp .env.example .env
   ```

3. **Start the training pipeline**
   ```bash
   docker compose up train
   ```
   This runs the full pipeline: ingest → validate → build features → train → evaluate → register model. It will take several minutes on CPU.

4. **Start the inference API**
   Once training finishes, start the `serve` service in the background:
   ```bash
   docker compose up -d serve
   ```

5. **Verify the health endpoint**
   ```bash
   curl http://localhost:8080/health
   ```
   Expected response:
   ```json
   {"status":"healthy","model_loaded":true}
   ```

6. **Test the prediction endpoint**
   ```bash
   curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"code": "#include <stdio.h>\nint main() {\n    int n;\n    scanf(\"%d\", &n);\n    int sum = 0;\n    for (int i = 1; i <= n; i++) {\n        sum += i;\n    }\n    printf(\"%d\\n\", sum);\n    return 0;\n}"}'
   ```
   Expected response:
   ```json
   {
     "prediction": "human",
     "human_probability": 0.85,
     "ai_probability": 0.15
   }
   ```

7. **View logs**
   ```bash
   docker compose logs -f serve
   ```

8. **Stop services when done**
   ```bash
   docker compose down
   ```

**Optional services:**
- **Monitor drift:**
  ```bash
  docker compose run --rm monitor
  ```
- **Retrain pipeline:**
  ```bash
  docker compose run --rm retrain
  ```

### Option 3: Jupyter notebook

```bash
jupyter lab GraphCodeBERT_Course_Detect_Human_and_AI_code.ipynb
```

## Model files

| File | Description |
|------|-------------|
| `models/checkpoints/graphcodebert_human_ai.pt` | Fine-tuned PyTorch checkpoint |
| `models/onnx/graphcodebert_human_ai_fp16.onnx` | ONNX FP16 model for production |
| `tokenizer/` | Local GraphCodeBERT tokenizer |

## Architecture

```
C source code
    ↓
Tokenizer (microsoft/graphcodebert-base)
    ↓
input_ids + attention_mask
    ↓
GraphCodeBERT encoder (frozen)
    ↓
CLS embedding (768-d)
    ↓
Linear(768 → 2) + Dropout
    ↓
Logits
    ↓
Sigmoid (at inference)
    ↓
Human / AI probabilities
```

## Predicted files

- `Makefile` — common commands for pipeline stages
- `docker-compose.yml` — containerized train, serve, monitor, retrain services
- `docker/Dockerfile` — production inference container
- `.github/workflows/ci-cd.yml` — GitHub Actions CI/CD pipeline
- `config/config.yaml` — pipeline hyperparameters and settings
- `config/schema.json` — dataset validation schema
- `requirements.txt` — Python dependencies
- `pyproject.toml` — project configuration
- `.env.example` — environment variable template
- `notebooks/GraphCodeBERT_Course_Detect_Human_and_AI_code.ipynb` — main training and inference notebook
- `dataset.zip` — course dataset with human and AI C files

## Notes

- GraphCodeBERT is best suited for C/C++/Java/Python. This project exercises it on C code.
- The tokenizer is saved locally to `tokenizer/` during feature engineering for fully offline inference.
- The ONNX model is standalone for the neural network path. Only the tokenizer is needed alongside it for production.

## MLOps Pipeline

This project includes a production-grade MLOps pipeline with the following stages:

```
Data Ingestion → Data Validation → Feature Engineering → Model Training →
Model Evaluation → Model Registry → Deployment → Monitoring → Retraining
```

### Directory Structure

```
human-ai-code-detect/
├── .github/workflows/
│   └── ci-cd.yml          # GitHub Actions CI/CD pipeline
├── config/
│   ├── config.yaml        # Pipeline hyperparameters and settings
│   └── schema.json        # Dataset validation schema
├── data/
│   ├── raw/               # Raw ingested data (gitignored)
│   ├── processed/         # Validated, tokenized data and artifacts (gitignored)
│   └── external/          # External datasets
├── docker/
│   └── Dockerfile         # Production inference container
├── models/
│   ├── checkpoints/       # PyTorch training checkpoints (gitignored)
│   ├── onnx/              # ONNX exported models (gitignored)
│   └── registry/          # MLflow tracking data (gitignored)
├── notebooks/             # Study and exploration notebooks
├── src/
│   ├── data/              # Data ingestion and validation
│   │   ├── ingest.py      # Extract dataset from archive
│   │   └── validate.py    # Schema and quality validation
│   ├── features/          # Feature engineering
│   │   └── build_features.py  # Tokenization and embedding
│   ├── models/            # Training, evaluation, registry
│   │   ├── train.py       # Model training loop
│   │   ├── evaluate.py    # Model evaluation metrics
│   │   └── registry.py    # MLflow model registry
│   ├── inference/         # Serving and prediction
│   │   ├── api.py         # FastAPI inference server
│   │   └── predict.py     # ONNX Runtime prediction class
│   ├── monitoring/        # Drift detection and logging
│   │   ├── drift.py       # Data drift detection
│   │   └── log_predictions.py  # Prediction audit logging
│   └── retrain/           # Automated retraining pipeline
│       └── pipeline.py    # End-to-end retraining orchestration
├── tests/                 # Unit tests
├── .env.example           # Environment variable template
├── Makefile               # Common commands
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
└── docker-compose.yml     # Container orchestration
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| Data Ingestion | `src/data/ingest.py` | Extracts `dataset.zip` into `data/raw/` |
| Data Validation | `src/data/validate.py` | Validates schema, class balance, file sizes |
| Feature Engineering | `src/features/build_features.py` | Tokenizes C files with GraphCodeBERT tokenizer |
| Model Training | `src/models/train.py` | Fine-tunes GraphCodeBERT with frozen encoder |
| Model Evaluation | `src/models/evaluate.py` | Computes accuracy, F1, precision, recall, AUC-ROC |
| Model Registry | `src/models/registry.py` | Registers model in MLflow with staging promotion |
| Deployment | `src/inference/api.py` | FastAPI server with `/predict` and `/health` endpoints |
| Monitoring | `src/monitoring/drift.py` | Detects data drift using embedding statistics |
| Retraining | `src/retrain/pipeline.py` | Orchestrates full pipeline re-run |

### Running the Pipeline

```bash
# Install dependencies
make install

# Run data ingestion and validation
make data

# Run feature engineering
make features

# Train and export model
make train

# Evaluate the model
make evaluate

# Register model in MLflow
make registry

# Start the inference API
make deploy

# Run the retraining pipeline
make retrain

# Run tests
make test

# Lint
make lint

# Clean generated artifacts
make clean
```

### Free Tools Used

| Purpose | Tool | License |
|---------|------|---------|
| Experiment tracking | MLflow | Open-source (BSD) |
| Model registry | MLflow Model Registry | Open-source (BSD) |
| API serving | FastAPI + Uvicorn | Open-source (MIT) |
| Inference | ONNX Runtime | Open-source (MIT) |
| Containerization | Docker | Open-source (Apache 2.0) |
| CI/CD | GitHub Actions | Free tier |
| Data validation | Custom (JSON Schema) | — |
| Monitoring | Custom (drift detection) | — |
| Orchestration | Makefile + Python scripts | — |
