import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_mean_shift_detects_clear_change():
    result = DataAnalysisEngine().analyze(
        "mean_shift_change_point",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
            "t9",
            "t10",
        ],
        values=[
            0,
            0,
            0,
            0,
            0,
            10,
            10,
            10,
            10,
            10,
        ],
    )

    assert result.result[
        "detected"
    ] is True

    assert result.result[
        "change_index"
    ] == 5

    assert result.result[
        "change_timestamp"
    ] == "t6"


def test_mean_shift_constant_series_has_no_change():
    result = DataAnalysisEngine().analyze(
        "mean_shift_change_point",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
        ],
        values=[
            5,
            5,
            5,
            5,
            5,
            5,
        ],
    )

    assert result.result[
        "detected"
    ] is False

    assert result.result[
        "change_index"
    ] is None


def test_mean_shift_threshold_is_configurable():
    result = DataAnalysisEngine().analyze(
        "mean_shift_change_point",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
            "t8",
        ],
        values=[
            0,
            0,
            0,
            0,
            10,
            10,
            10,
            10,
        ],
        threshold=100.0,
    )

    assert result.result[
        "detected"
    ] is False


def test_mean_shift_rejects_insufficient_series():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "mean_shift_change_point",
            time=[
                "t1",
                "t2",
                "t3",
            ],
            values=[
                1,
                1,
                10,
            ],
            min_segment_size=2,
        )
