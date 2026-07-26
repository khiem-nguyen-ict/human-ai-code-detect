import json
import logging
import os
import sys

from pathlib import Path

from dotenv import load_dotenv

from src.models.registry import should_deploy, _compute_deployment_score, get_production_metrics

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> None:
    metrics_path = os.getenv("TEST_METRICS_PATH", "./data/processed/test_metrics.json")

    if not Path(metrics_path).exists():
        raise FileNotFoundError(f"Test metrics not found: {metrics_path}")

    with open(metrics_path) as f:
        new_metrics = json.load(f)

    deploy, reason = should_deploy(new_metrics)

    new_score = _compute_deployment_score(new_metrics)
    prod_metrics = get_production_metrics()
    prev_score = _compute_deployment_score(prod_metrics) if prod_metrics else None

    logger.info(
        "Deployment gate - F1: %.4f, Precision: %.4f, Recall: %.4f, Score: %.4f",
        new_metrics.get("f1", 0.0),
        new_metrics.get("precision", 0.0),
        new_metrics.get("recall", 0.0),
        new_score,
    )
    if prev_score is not None:
        logger.info("Production model score: %.4f", prev_score)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"should_deploy={str(deploy).lower()}\n")
            f.write(f"deploy_reason={reason}\n")

    if deploy:
        logger.info("Deployment gate PASSED: %s", reason)
    else:
        logger.info("Deployment gate SKIPPED: %s", reason)


if __name__ == "__main__":
    main()
