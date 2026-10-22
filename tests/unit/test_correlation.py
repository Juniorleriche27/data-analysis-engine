import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.statistics.correlation import (
    PearsonCorrelation,
)


def test_pearson_perfect_positive_relationship():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "pearson_correlation",
        x=[1, 2, 3, 4, 5],
        y=[2, 4, 6, 8, 10],
    )

    assert result.result["coefficient"] == pytest.approx(1.0)
    assert result.result["sample_size"] == 5
    assert result.result["inference"] is None


def test_pearson_perfect_negative_relationship():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "pearson_correlation",
        x=[1, 2, 3, 4, 5],
        y=[10, 8, 6, 4, 2],
    )

    assert result.result["coefficient"] == pytest.approx(-1.0)


def test_spearman_detects_monotonic_nonlinear_relationship():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "spearman_correlation",
        x=[1, 2, 3, 4, 5],
        y=[1, 4, 9, 16, 25],
    )

    assert result.result["coefficient"] == pytest.approx(1.0)


def test_constant_input_is_rejected():
    method = PearsonCorrelation()

    with pytest.raises(InvalidAnalysisInputError):
        method.execute(
            x=[1, 1, 1, 1],
            y=[1, 2, 3, 4],
        )
