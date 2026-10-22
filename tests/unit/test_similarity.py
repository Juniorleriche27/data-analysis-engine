import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_euclidean_distance():
    result = DataAnalysisEngine().analyze(
        "euclidean_distance",
        x=[0, 0],
        y=[3, 4],
    )

    assert (
        result.result["distance"]
        == pytest.approx(5.0)
    )


def test_manhattan_distance():
    result = DataAnalysisEngine().analyze(
        "manhattan_distance",
        x=[0, 0],
        y=[3, 4],
    )

    assert (
        result.result["distance"]
        == pytest.approx(7.0)
    )


def test_minkowski_distance():
    result = DataAnalysisEngine().analyze(
        "minkowski_distance",
        x=[0, 0],
        y=[3, 4],
        p=3,
    )

    assert (
        result.result["distance"]
        == pytest.approx(
            91 ** (1 / 3)
        )
    )

    assert result.result["p"] == 3.0


def test_minkowski_rejects_invalid_p():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "minkowski_distance",
            x=[1, 2],
            y=[3, 4],
            p=0.5,
        )


def test_cosine_similarity_same_direction():
    result = DataAnalysisEngine().analyze(
        "cosine_similarity",
        x=[1, 2, 3],
        y=[2, 4, 6],
    )

    assert (
        result.result["similarity"]
        == pytest.approx(1.0)
    )


def test_cosine_distance_for_orthogonal_vectors():
    result = DataAnalysisEngine().analyze(
        "cosine_distance",
        x=[1, 0],
        y=[0, 1],
    )

    assert (
        result.result["similarity"]
        == pytest.approx(0.0)
    )

    assert (
        result.result["distance"]
        == pytest.approx(1.0)
    )


def test_cosine_rejects_zero_vector():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "cosine_similarity",
            x=[0, 0, 0],
            y=[1, 2, 3],
        )
