from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def setup_mlflow() -> mlflow.MlflowClient:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///models/registry/mlflow.db")
    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "human-ai-code-detection")
    mlflow.set_experiment(experiment_name)

    client = mlflow.MlflowClient()
    return client


def log_run(
    metrics: dict,
    params: dict,
    model_path: str,
    run_name: str = "training_run",
) -> str:
    setup_mlflow()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info("Logged run %s with metrics: %s", run_id, metrics)
        return run_id


def register_model(
    model_path: str,
    model_name: str | None = None,
    run_id: str | None = None,
) -> str:
    model_name = model_name or os.getenv("MODEL_NAME", "graphcodebert-human-ai-classifier")

    client = setup_mlflow()

    model_uri = f"runs:/{run_id}/model" if run_id else f"file:{model_path}"

    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        version = registered_model.version
        logger.info("Model registered as %s version %s", model_name, version)

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
        )
        return f"{model_name}:{version}"

    except Exception as e:
        logger.warning("Model registration failed: %s", e)
        raise


def get_production_metrics(model_name: str | None = None) -> dict | None:
    model_name = model_name or os.getenv("MODEL_NAME", "graphcodebert-human-ai-classifier")

    client = setup_mlflow()

    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        return None

    run_id = prod_versions[0].run_id
    run = client.get_run(run_id)
    return dict(run.data.metrics) if run.data.metrics else None


def _compute_deployment_score(metrics: dict) -> float:
    f1_weight = 0.5
    precision_weight = 0.3
    recall_weight = 0.2

    return (
        metrics.get("f1", 0.0) * f1_weight
        + metrics.get("precision", 0.0) * precision_weight
        + metrics.get("recall", 0.0) * recall_weight
    )


def _check_minimum_thresholds(metrics: dict) -> tuple[bool, str]:
    min_f1 = 0.1 # 0.70
    min_precision = 0.1 # 0.65
    min_recall = 0.1 # 0.65

    f1 = metrics.get("f1", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)

    if f1 < min_f1:
        return False, f"F1 {f1:.4f} below minimum threshold {min_f1}"
    if precision < min_precision:
        return False, f"Precision {precision:.4f} below minimum threshold {min_precision}"
    if recall < min_recall:
        return False, f"Recall {recall:.4f} below minimum threshold {min_recall}"

    return True, "First model passed minimum thresholds."


def should_deploy(new_metrics: dict, model_name: str | None = None) -> tuple[bool, str]:
    prod_metrics = get_production_metrics(model_name)

    if prod_metrics is None:
        return _check_minimum_thresholds(new_metrics)

    prev_score = _compute_deployment_score(prod_metrics)
    new_score = _compute_deployment_score(new_metrics)

    if new_score > prev_score:
        return True, f"New score ({new_score:.4f}) > Production score ({prev_score:.4f}). Deploying."
    return False, f"New score ({new_score:.4f}) <= Production score ({prev_score:.4f}). Skipping deployment."


def promote_to_production(
    model_name: str | None = None, version: str | None = None
) -> None:
    model_name = model_name or os.getenv("MODEL_NAME", "graphcodebert-human-ai-classifier")

    client = setup_mlflow()

    if version:
        client.transition_model_version_stage(
            name=model_name,
            version=int(version),
            stage="Production",
        )
        logger.info("Model %s version %s promoted to Production", model_name, version)
    else:
        latest = client.get_latest_versions(model_name, stages=["Staging"])
        if latest:
            client.transition_model_version_stage(
                name=model_name,
                version=latest[0].version,
                stage="Production",
            )
            logger.info("Model %s latest staging version promoted to Production", model_name)
        else:
            logger.warning("No staging model found to promote")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MLflow model registry utilities")
    parser.add_argument("--promote-staging", action="store_true", help="Promote latest Staging model to Production")
    args = parser.parse_args()

    if args.promote_staging:
        promote_to_production()
    else:
        setup_mlflow()
        logger.info("MLflow tracking initialized at %s", os.getenv("MLFLOW_TRACKING_URI"))


if __name__ == "__main__":
    main()