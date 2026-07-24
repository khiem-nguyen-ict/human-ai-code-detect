import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def load_config(config_path: str = "./config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_data_path(data_path: Path, human_dir: str, ai_dir: str) -> Path:
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    if (data_path / human_dir).exists() and (data_path / ai_dir).exists():
        return data_path

    subdirs = [
        d for d in data_path.iterdir()
        if d.is_dir() and d.name not in ("__MACOSX",)
    ]
    if len(subdirs) == 1:
        candidate = subdirs[0]
        if (candidate / human_dir).exists() and (candidate / ai_dir).exists():
            logger.info("Found nested dataset directory: %s", candidate)
            return candidate

    return data_path


def validate_dataset(
    data_dir: str = "./data/raw",
    config_path: str = "./config/config.yaml",
) -> dict:
    config = load_config(config_path)
    data_path = _resolve_data_path(
        Path(data_dir),
        config.get("data", {}).get("human_dir", "human"),
        config.get("data", {}).get("ai_dir", "ai"),
    )

    human_dir_name = config.get("data", {}).get("human_dir", "human")
    ai_dir_name = config.get("data", {}).get("ai_dir", "ai")

    human_dir = data_path / human_dir_name
    ai_dir = data_path / ai_dir_name
    extension = config.get("data", {}).get("file_extension", ".c")
    min_files = config.get("data", {}).get("min_files_per_class", 10)
    max_size_mb = config.get("data", {}).get("max_file_size_mb", 10.0)

    report = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "human": {"file_count": 0, "total_size_mb": 0.0, "files": []},
        "ai": {"file_count": 0, "total_size_mb": 0.0, "files": []},
    }

    for label, directory in [("human", human_dir), ("ai", ai_dir)]:
        if not directory.exists():
            report["valid"] = False
            report["errors"].append(f"Missing directory: {directory}")
            continue

        files = list(directory.rglob(f"*{extension}"))
        report[label]["file_count"] = len(files)

        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            report[label]["total_size_mb"] += size_mb
            report[label]["files"].append(str(f))

            if size_mb > max_size_mb:
                report["warnings"].append(
                    f"{f} exceeds max file size ({size_mb:.2f} MB > {max_size_mb} MB)"
                )

        if report[label]["file_count"] < min_files:
            report["valid"] = False
            report["errors"].append(
                f"{label} class has {report[label]['file_count']} files, "
                f"minimum required is {min_files}"
            )

    human_count = report["human"]["file_count"]
    ai_count = report["ai"]["file_count"]
    total = human_count + ai_count

    if total == 0:
        report["valid"] = False
        report["errors"].append("No source files found in dataset")
    else:
        ratio = human_count / total if total > 0 else 0
        logger.info(
            "Class balance: human=%.1f%%, ai=%.1f%%", ratio * 100, (1 - ratio) * 100
        )
        if ratio < 0.2 or ratio > 0.8:
            report["warnings"].append(
                f"Class imbalance detected: human={ratio:.2%}, ai={1-ratio:.2%}"
            )

    report["total_files"] = total
    report["total_size_mb"] = round(
        report["human"]["total_size_mb"] + report["ai"]["total_size_mb"], 2
    )

    if report["errors"]:
        logger.error("Validation failed: %s", report["errors"])
    else:
        logger.info("Validation passed. %d files, %.2f MB total.", total, report["total_size_mb"])

    return report


def save_report(report: dict, output_path: str = "./data/processed/validation_report.json") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to %s", output_path)


def main() -> None:
    raw_dir = os.getenv("RAW_DATA_DIR", "./data/raw")
    processed_dir = os.getenv("PROCESSED_DATA_DIR", "./data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    report = validate_dataset(raw_dir)
    save_report(report)

    if not report["valid"]:
        raise RuntimeError("Dataset validation failed. See validation_report.json for details.")


if __name__ == "__main__":
    main()