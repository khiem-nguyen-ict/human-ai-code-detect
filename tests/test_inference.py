

def test_drift_import() -> None:
    from src.monitoring.drift import detect_drift, save_embeddings
    assert detect_drift is not None
    assert save_embeddings is not None


def test_log_predictions_import() -> None:
    from src.monitoring.log_predictions import get_prediction_history, log_prediction
    assert log_prediction is not None
    assert get_prediction_history is not None


def test_retrain_pipeline_import() -> None:
    from src.retrain.pipeline import run_pipeline, run_stage
    assert run_pipeline is not None
    assert run_stage is not None


def test_ingest_import() -> None:
    from src.data.ingest import ingest_dataset, list_source_files
    assert ingest_dataset is not None
    assert list_source_files is not None


def test_features_import() -> None:
    from src.features.build_features import build_dataset, load_tokenizer
    assert load_tokenizer is not None
    assert build_dataset is not None