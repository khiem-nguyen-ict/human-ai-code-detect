import logging
import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def ingest_dataset(
    zip_path: str = "./dataset.zip",
    output_dir: str = "./data/raw",
) -> Path:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset archive not found at {zip_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting dataset from %s to %s", zip_path, output)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output)

    extracted_dirs = list(output.iterdir())
    if not extracted_dirs:
        raise RuntimeError(f"No contents found after extracting {zip_path}")

    for d in extracted_dirs:
        logger.info("Extracted: %s", d)

    return output


def _resolve_data_path(data_path: Path) -> Path:
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    if (data_path / "human").exists() and (data_path / "ai").exists():
        return data_path

    subdirs = [
        d for d in data_path.iterdir()
        if d.is_dir() and d.name not in ("__MACOSX",)
    ]
    if len(subdirs) == 1:
        candidate = subdirs[0]
        if (candidate / "human").exists() and (candidate / "ai").exists():
            logger.info("Found nested dataset directory: %s", candidate)
            return candidate

    return data_path


def list_source_files(data_dir: str, extension: str = ".c") -> dict[str, list[str]]:
    data_path = _resolve_data_path(Path(data_dir))

    result = {}
    for split_name in ["human", "ai"]:
        split_dir = data_path / split_name
        if split_dir.exists():
            files = sorted(
                str(f) for f in split_dir.rglob(f"*{extension}") if f.is_file()
            )
            result[split_name] = files
            logger.info("Found %d %s files", len(files), split_name)
        else:
            logger.warning("No %s directory found in %s", split_name, data_path)
            result[split_name] = []

    return result


def main() -> None:
    raw_dir = os.getenv("RAW_DATA_DIR", "./data/raw")
    zip_path = "./dataset.zip"

    if os.path.exists(zip_path):
        ingest_dataset(zip_path, raw_dir)
    else:
        logger.warning("No dataset.zip found at %s. Using existing raw data.", zip_path)

    files = list_source_files(raw_dir)
    total = sum(len(v) for v in files.values())
    logger.info("Ingestion complete. Total files: %d", total)


if __name__ == "__main__":
    main()