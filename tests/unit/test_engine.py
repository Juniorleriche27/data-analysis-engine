import pytest

from data_analysis_engine import DataAnalysisEngine


def test_descriptive_statistics():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "descriptive_statistics",
        values=[10, 20, 30, 40, 50],
    )

    assert result.status == "success"
    assert result.category == "statistics"
    assert result.result["count"] == 5
    assert result.result["mean"] == pytest.approx(30.0)
    assert result.result["median"] == pytest.approx(30.0)
    assert result.result["variance_population"] == pytest.approx(200.0)
    assert result.result["minimum"] == 10
    assert result.result["maximum"] == 50


def test_method_is_registered():
    engine = DataAnalysisEngine()

    names = {
        method["name"]
        for method in engine.available_methods()
    }

    assert "descriptive_statistics" in names
