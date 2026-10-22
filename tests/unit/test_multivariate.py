import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_covariance_matrix():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "covariance_matrix",
        data={
            "x": [1, 2, 3],
            "y": [2, 4, 6],
        },
    )

    matrix = result.result["covariance_matrix"]

    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(2.0)
    assert matrix[1][0] == pytest.approx(2.0)
    assert matrix[1][1] == pytest.approx(4.0)

    assert result.result["ddof"] == 1


def test_linear_combination_variance_matches_formula():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "linear_combination_variance",
        data={
            "x": [1, 2, 3],
            "y": [2, 4, 6],
        },
        weights={
            "x": 0.5,
            "y": 0.5,
        },
    )

    assert result.result["variance"] == pytest.approx(2.25)
    assert result.result["standard_deviation"] == pytest.approx(1.5)


def test_variance_contributions_sum_to_total_variance():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "linear_combination_variance",
        data={
            "x": [1, 2, 3],
            "y": [2, 4, 6],
        },
        weights={
            "x": 0.5,
            "y": 0.5,
        },
    )

    contributions = result.result["contributions"]

    total = sum(
        item["variance_contribution"]
        for item in contributions.values()
    )

    assert total == pytest.approx(
        result.result["variance"]
    )

    assert (
        contributions["x"]["contribution_percentage"]
        == pytest.approx(1 / 3)
    )

    assert (
        contributions["y"]["contribution_percentage"]
        == pytest.approx(2 / 3)
    )


def test_weights_do_not_need_to_sum_to_one():
    engine = DataAnalysisEngine()

    result = engine.analyze(
        "linear_combination_variance",
        data={
            "x": [1, 2, 4, 7],
            "y": [2, 1, 5, 3],
        },
        weights={
            "x": 2.0,
            "y": -0.5,
        },
    )

    assert result.result["variance"] >= 0


def test_weight_columns_must_match_data_columns():
    engine = DataAnalysisEngine()

    with pytest.raises(InvalidAnalysisInputError):
        engine.analyze(
            "linear_combination_variance",
            data={
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            },
            weights={
                "x": 0.5,
            },
        )


def test_multivariate_rejects_non_numeric_data():
    engine = DataAnalysisEngine()

    with pytest.raises(InvalidAnalysisInputError):
        engine.analyze(
            "covariance_matrix",
            data={
                "x": [1, 2, 3],
                "y": ["a", "b", "c"],
            },
        )
