import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_pca_perfectly_collinear_variables():
    result = DataAnalysisEngine().analyze(
        "principal_component_analysis",
        data={
            "x": [1, 2, 3, 4],
            "y": [2, 4, 6, 8],
        },
    )

    assert result.result["standardized"] is True

    assert (
        result.result[
            "explained_variance_ratio"
        ][0]
        == pytest.approx(1.0)
    )

    assert (
        result.result[
            "explained_variance_ratio"
        ][1]
        == pytest.approx(0.0)
    )


def test_pca_explained_variance_sums_to_one():
    result = DataAnalysisEngine().analyze(
        "principal_component_analysis",
        data={
            "x": [1, 2, 4, 7, 8],
            "y": [2, 1, 5, 3, 9],
            "z": [5, 4, 2, 8, 6],
        },
    )

    ratios = result.result[
        "explained_variance_ratio"
    ]

    assert sum(ratios) == pytest.approx(
        1.0
    )


def test_pca_can_limit_number_of_components():
    result = DataAnalysisEngine().analyze(
        "principal_component_analysis",
        data={
            "x": [1, 2, 3, 4],
            "y": [2, 5, 6, 9],
            "z": [7, 4, 8, 3],
        },
        n_components=1,
    )

    assert result.result[
        "n_components"
    ] == 1

    assert len(
        result.result["components"]
    ) == 1

    assert all(
        set(score.keys()) == {"PC1"}
        for score in result.result["scores"]
    )


def test_pca_can_use_covariance_without_standardization():
    result = DataAnalysisEngine().analyze(
        "principal_component_analysis",
        data={
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
        },
        standardize=False,
    )

    assert result.result[
        "standardized"
    ] is False

    assert result.result[
        "scales"
    ] == {
        "x": 1.0,
        "y": 1.0,
    }


def test_pca_rejects_invalid_component_count():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "principal_component_analysis",
            data={
                "x": [1, 2, 3, 4],
                "y": [2, 4, 6, 8],
            },
            n_components=0,
        )


def test_pca_scores_preserve_sample_count():
    result = DataAnalysisEngine().analyze(
        "principal_component_analysis",
        data={
            "x": [1, 2, 3, 4, 5],
            "y": [2, 3, 7, 8, 10],
        },
    )

    assert len(
        result.result["scores"]
    ) == 5
