from pathlib import Path


def test_config_exists() -> None:
    assert Path("./config/config.yaml").exists()


def test_schema_exists() -> None:
    assert Path("./config/schema.json").exists()


def test_gitignore_exists() -> None:
    assert Path("./.gitignore").exists()


def test_requirements_exists() -> None:
    assert Path("./requirements.txt").exists()


def test_makefile_exists() -> None:
    assert Path("./Makefile").exists()


def test_src_init_exists() -> None:
    assert Path("./src/__init__.py").exists()


def test_data_init_exists() -> None:
    assert Path("./src/data/__init__.py").exists()


def test_features_init_exists() -> None:
    assert Path("./src/features/__init__.py").exists()


def test_models_init_exists() -> None:
    assert Path("./src/models/__init__.py").exists()


def test_inference_init_exists() -> None:
    assert Path("./src/inference/__init__.py").exists()


def test_monitoring_init_exists() -> None:
    assert Path("./src/monitoring/__init__.py").exists()


def test_retrain_init_exists() -> None:
    assert Path("./src/retrain/__init__.py").exists()