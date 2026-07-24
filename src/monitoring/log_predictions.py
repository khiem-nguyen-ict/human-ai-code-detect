from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def log_prediction(
    code: str,
    prediction: str,
    human_prob: float,
    ai_prob: float,
    log_path: str | None = None,
) -> None:
    log_path = log_path or os.getenv(
        "PREDICTION_LOG_PATH", "./data/processed/predictions_log.csv"
    )

    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_preview": code[:200],
        "prediction": prediction,
        "human_probability": human_prob,
        "ai_probability": ai_prob,
    }

    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Prediction logged: %s (human=%.4f, ai=%.4f)", prediction, human_prob, ai_prob)


def get_prediction_history(
    log_path: str | None = None,
    limit: int = 100,
) -> list[dict]:
    log_path = log_path or os.getenv(
        "PREDICTION_LOG_PATH", "./data/processed/predictions_log.csv"
    )

    if not Path(log_path).exists():
        return []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return rows[-limit:]


def main() -> None:
    from src.inference.predict import Predictor

    predictor = Predictor()

    sample = "#include <stdio.h>\nint main() { return 0; }"
    result = predictor.predict(sample)
    log_prediction(
        code=sample,
        prediction=result["prediction"],
        human_prob=result["human_probability"],
        ai_prob=result["ai_probability"],
    )

    history = get_prediction_history()
    logger.info("Total logged predictions: %d", len(history))


if __name__ == "__main__":
    main()