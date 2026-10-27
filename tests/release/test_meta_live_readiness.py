import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.integrations.meta_live_smoke import (
    MetaLiveSmokeError,
    build_demo_dataframe,
    require_model_api_key,
)


def test_live_smoke_requires_model_api_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    with pytest.raises(MetaLiveSmokeError):
        require_model_api_key()


def test_live_smoke_demo_dataset_is_deterministic():
    dataframe = build_demo_dataframe()

    assert dataframe["sales"].tolist() == [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]

    assert dataframe["sales"].median() == pytest.approx(30.0)


def test_live_readiness_does_not_change_native_registry():
    methods = DataAnalysisEngine().available_methods()

    assert len(methods) == len({method["name"] for method in methods})
