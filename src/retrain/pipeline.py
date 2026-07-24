import logging
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def run_stage(stage_name: str, command: list[str]) -> bool:
    logger.info("Running stage: %s", stage_name)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        logger.info("%s stdout:\n%s", stage_name, result.stdout)
    if result.stderr:
        logger.error("%s stderr:\n%s", stage_name, result.stderr)

    success = result.returncode == 0
    if success:
        logger.info("Stage '%s' completed successfully", stage_name)
    else:
        logger.error("Stage '%s' failed with return code %d", stage_name, result.returncode)

    return success


def run_pipeline() -> bool:
    stages = [
        ("data_ingestion", [sys.executable, "-m", "src.data.ingest"]),
        ("data_validation", [sys.executable, "-m", "src.data.validate"]),
        ("feature_engineering", [sys.executable, "-m", "src.features.build_features"]),
        ("model_training", [sys.executable, "-m", "src.models.train"]),
        ("model_evaluation", [sys.executable, "-m", "src.models.evaluate"]),
        ("model_registry", [sys.executable, "-m", "src.models.registry"]),
    ]

    for stage_name, command in stages:
        success = run_stage(stage_name, command)
        if not success:
            logger.error("Pipeline stopped at stage: %s", stage_name)
            return False

    logger.info("All pipeline stages completed successfully")
    return True


def main() -> None:
    success = run_pipeline()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()