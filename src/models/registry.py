import logging
import os
from pathlib import Path
from typing import Optional

import mlflow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def setup_mlflow() -> mlflow.MlflowClient:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./models/registry/mlruns")
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
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
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


def promote_to_production(
    model_name: Optional[str] = None, version: Optional[str] = None
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
    setup_mlflow()
    logger.info("MLflow tracking initialized at %s", os.getenv("MLFLOW_TRACKING_URI"))


if __name__ == "__main__":
    main()