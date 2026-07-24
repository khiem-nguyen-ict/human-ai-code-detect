import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def ingest_dataset(
    zip_path: str = "./dataset.zip",
    output_dir: str = "./data/raw",
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(zip_path):
        logger.warning("Dataset archive not found at %s. Skipping extraction.", zip_path)
        return output

    logger.info("Extracting dataset from %s to %s", zip_path, output)
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output)

    for d in output.iterdir():
        logger.info("Extracted: %s", d)

    return output


def list_source_files(data_dir: str, extension: str = ".c") -> dict[str, list[str]]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    if (data_path / "human").exists() and (data_path / "ai").exists():
        resolved = data_path
    else:
        subdirs = [
            d for d in data_path.iterdir()
            if d.is_dir() and d.name not in ("__MACOSX",)
        ]
        if len(subdirs) == 1 and (subdirs[0] / "human").exists() and (subdirs[0] / "ai").exists():
            resolved = subdirs[0]
        else:
            resolved = data_path

    result = {}
    for split_name in ["human", "ai"]:
        split_dir = resolved / split_name
        if split_dir.exists():
            files = sorted(
                str(f) for f in split_dir.rglob(f"*{extension}") if f.is_file()
            )
            result[split_name] = files
            logger.info("Found %d %s files", len(files), split_name)
        else:
            logger.warning("No %s directory found in %s", split_name, resolved)
            result[split_name] = []

    return result


def main() -> None:
    raw_dir = os.getenv("RAW_DATA_DIR", "./data/raw")
    files = list_source_files(raw_dir)
    total = sum(len(v) for v in files.values())
    logger.info("Ingestion complete. Total files: %d", total)


if __name__ == "__main__":
    main()