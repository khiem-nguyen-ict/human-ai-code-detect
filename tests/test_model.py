

def test_predictor_import() -> None:
    from src.inference.predict import Predictor
    assert Predictor is not None


def test_predictor_class_has_predict_method() -> None:
    from src.inference.predict import Predictor
    assert hasattr(Predictor, "predict")
    assert callable(Predictor.predict)


def test_api_import() -> None:
    from src.inference.api import app
    assert app is not None


def test_app_routes_exist() -> None:
    from src.inference.api import app
    routes = [r.name for r in app.routes]
    assert "health_check" in routes
    assert "predict" in routes


def test_classification_head_import() -> None:
    from src.models.train import ClassificationHead
    assert ClassificationHead is not None


def test_code_dataset_import() -> None:
    from src.models.train import CodeDataset
    assert CodeDataset is not None