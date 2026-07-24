import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def compute_embedding_stats(embeddings: np.ndarray) -> dict:
    return {
        "mean": np.mean(embeddings, axis=0).tolist(),
        "std": np.std(embeddings, axis=0).tolist(),
        "min": np.min(embeddings, axis=0).tolist(),
        "max": np.max(embeddings, axis=0).tolist(),
        "shape": embeddings.shape,
    }


def detect_drift(
    reference_path: str = "./data/processed/reference_embeddings.npy",
    current_path: str = "./data/processed/current_embeddings.npy",
    threshold: float = 0.05,
) -> dict:
    if not Path(reference_path).exists():
        logger.warning("No reference embeddings found at %s", reference_path)
        return {"drift_detected": False, "reason": "No reference data"}

    if not Path(current_path).exists():
        logger.warning("No current embeddings found at %s", current_path)
        return {"drift_detected": False, "reason": "No current data"}

    reference = np.load(reference_path)
    current = np.load(current_path)

    if reference.shape != current.shape:
        return {
            "drift_detected": True,
            "reason": f"Shape mismatch: ref={reference.shape}, current={current.shape}",
        }

    ref_mean = np.mean(reference, axis=0)
    cur_mean = np.mean(current, axis=0)
    ref_std = np.std(reference, axis=0) + 1e-8

    relative_shift = np.abs(cur_mean - ref_mean) / ref_std
    max_shift = float(np.max(relative_shift))
    mean_shift = float(np.mean(relative_shift))

    drift_detected = max_shift > threshold or mean_shift > threshold / 2

    result = {
        "drift_detected": drift_detected,
        "max_relative_shift": max_shift,
        "mean_relative_shift": mean_shift,
        "threshold": threshold,
    }

    if drift_detected:
        logger.warning("Data drift detected: max_shift=%.4f, mean_shift=%.4f", max_shift, mean_shift)
    else:
        logger.info("No significant drift detected: max_shift=%.4f", max_shift)

    return result


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    logger.info("Embeddings saved to %s", path)


def main() -> None:
    threshold = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
    result = detect_drift(threshold=threshold)

    report_path = "./data/processed/drift_report.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["drift_detected"]:
        logger.warning("Drift report saved to %s", report_path)
    else:
        logger.info("Drift report saved to %s", report_path)


if __name__ == "__main__":
    main()