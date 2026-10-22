import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_dtw_identical_series_have_zero_distance():
    result = DataAnalysisEngine().analyze(
        "dynamic_time_warping",
        x=[1, 2, 3, 4],
        y=[1, 2, 3, 4],
    )

    assert result.result[
        "distance"
    ] == pytest.approx(0.0)


def test_dtw_supports_different_sequence_lengths():
    result = DataAnalysisEngine().analyze(
        "dynamic_time_warping",
        x=[1, 2, 3],
        y=[1, 1, 2, 3],
    )

    assert result.result[
        "distance"
    ] == pytest.approx(0.0)

    assert result.result[
        "x_length"
    ] == 3

    assert result.result[
        "y_length"
    ] == 4


def test_dtw_returns_alignment_path():
    result = DataAnalysisEngine().analyze(
        "dynamic_time_warping",
        x=[1, 2, 3],
        y=[1, 2, 3],
    )

    path = result.result[
        "warping_path"
    ]

    assert path[0] == [0, 0]
    assert path[-1] == [2, 2]


def test_dtw_rejects_window_too_small():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "dynamic_time_warping",
            x=[1, 2, 3],
            y=[1, 1, 2, 3, 4],
            window=1,
        )
