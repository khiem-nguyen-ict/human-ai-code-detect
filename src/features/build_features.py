import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("BASE_MODEL", "microsoft/graphcodebert-base")
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))
TOKENIZER_DIR = os.getenv("TOKENIZER_PATH", "./tokenizer")


def load_tokenizer(save: bool = False) -> AutoTokenizer:
    logger.info("Loading tokenizer for %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if save:
        out = Path(TOKENIZER_DIR)
        out.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out))
        logger.info("Tokenizer saved to %s", out)

    return tokenizer


def tokenize_files(
    file_paths: list[str],
    tokenizer: AutoTokenizer,
) -> dict[str, np.ndarray]:
    texts = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            texts.append(f.read())

    logger.info("Tokenizing %d files", len(texts))
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )

    return {
        "input_ids": np.array(encoded["input_ids"]),
        "attention_mask": np.array(encoded["attention_mask"]),
    }


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


def build_dataset(
    data_dir: str = "./data/raw",
    tokenizer: AutoTokenizer = None,
    save: bool = True,
) -> dict[str, dict]:
    if tokenizer is None:
        tokenizer = load_tokenizer()

    data_path = _resolve_data_path(Path(data_dir))
    splits = {}

    for label_name, label_id in [("human", 0), ("ai", 1)]:
        label_dir = data_path / label_name
        if not label_dir.exists():
            logger.warning("Skipping missing directory: %s", label_dir)
            continue

        files = sorted(label_dir.rglob("*.c"))
        if not files:
            continue

        tokens = tokenize_files([str(f) for f in files], tokenizer)

        splits[label_name] = {
            "file_paths": [str(f) for f in files],
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": np.full(len(files), label_id, dtype=np.int64),
        }

        logger.info(
            "%s: %d files, input_ids shape %s",
            label_name,
            len(files),
            tokens["input_ids"].shape,
        )

    if save:
        out = Path("./data/processed")
        out.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            np.savez(
                out / f"{split_name}_tokenized.npz",
                input_ids=split_data["input_ids"],
                attention_mask=split_data["attention_mask"],
                labels=split_data["labels"],
                file_paths=split_data["file_paths"],
            )
            logger.info("Saved %s tokenized data to %s", split_name, out)

    return splits


def load_tokenized_data(
    split_name: str,
    data_dir: str = "./data/processed",
) -> dict:
    path = Path(data_dir) / f"{split_name}_tokenized.npz"
    if not path.exists():
        raise FileNotFoundError(f"Tokenized data not found: {path}")

    data = np.load(path, allow_pickle=True)
    return {
        "input_ids": data["input_ids"],
        "attention_mask": data["attention_mask"],
        "labels": data["labels"],
        "file_paths": data["file_paths"],
    }


def main() -> None:
    tokenizer = load_tokenizer(save=True)
    build_dataset("./data/raw", tokenizer=tokenizer, save=True)
    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()