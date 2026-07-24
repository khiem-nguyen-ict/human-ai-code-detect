import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uvicorn import Config, Server

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Human vs AI Code Detector",
    version="0.1.0",
    description="Detect whether C source code is human-written or AI-generated",
)

predictor = None


class PredictionRequest(BaseModel):
    code: str


class PredictionResponse(BaseModel):
    prediction: str
    human_probability: float
    ai_probability: float


@app.on_event("startup")
def load_model() -> None:
    global predictor
    from src.inference.predict import Predictor
    predictor = Predictor()
    logger.info("Model loaded successfully")


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = predictor.predict(request.code)
    return PredictionResponse(**result)


def main() -> None:
    host = os.getenv("DEPLOY_HOST", "0.0.0.0")
    port = int(os.getenv("DEPLOY_PORT", "8080"))
    workers = int(os.getenv("DEPLOY_WORKERS", "1"))

    logger.info("Starting server on %s:%d", host, port)
    config = Config(app=app, host=host, port=port, workers=workers)
    server = Server(config=config)
    server.run()


if __name__ == "__main__":
    main()