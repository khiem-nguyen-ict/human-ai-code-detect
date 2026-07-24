import json
from pathlib import Path

import pytest

from src.data.validate import load_config, save_report, validate_dataset


def test_load_config() -> None:
    config = load_config()
    assert "data" in config
    assert config["data"]["human_dir"] == "human"
    assert config["data"]["ai_dir"] == "ai"
    assert config["data"]["file_extension"] == ".c"


def test_validate_dataset_missing_dir() -> None:
    with pytest.raises(FileNotFoundError):
        validate_dataset(data_dir="./data/nonexistent")


def test_save_report_creates_file(tmp_path) -> None:
    report = {"valid": True, "errors": [], "warnings": [], "total_files": 0}
    report_path = tmp_path / "test_report.json"
    save_report(report, str(report_path))
    assert report_path.exists()

    with open(report_path, "r") as f:
        loaded = json.load(f)
    assert loaded["valid"] is True


def test_schema_json_is_valid() -> None:
    schema_path = Path("./config/schema.json")
    assert schema_path.exists()
    with open(schema_path, "r") as f:
        schema = json.load(f)
    assert "$schema" in schema
    assert schema["type"] == "object"