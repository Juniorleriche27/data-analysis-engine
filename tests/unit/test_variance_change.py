import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_variance_shift_detects_clear_change():
    result = DataAnalysisEngine().analyze(
        "variance_shift_change_point",
        time=[
            f"t{index}"
            for index in range(
                1,
                13,
            )
        ],
        values=[
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -10,
            10,
            -10,
            10,
            -10,
            10,
        ],
    )

    assert result.result[
        "detected"
    ] is True

    assert result.result[
        "change_index"
    ] == 6

    assert result.result[
        "change_timestamp"
    ] == "t7"

    assert result.result[
        "direction"
    ] == "variance_increase"


def test_variance_shift_constant_series_has_no_change():
    result = DataAnalysisEngine().analyze(
        "variance_shift_change_point",
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


def test_variance_shift_threshold_is_configurable():
    result = DataAnalysisEngine().analyze(
        "variance_shift_change_point",
        time=[
            f"t{index}"
            for index in range(
                1,
                13,
            )
        ],
        values=[
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -10,
            10,
            -10,
            10,
            -10,
            10,
        ],
        variance_ratio_threshold=1000.0,
    )

    assert result.result[
        "detected"
    ] is False


def test_variance_shift_rejects_insufficient_data():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "variance_shift_change_point",
            time=[
                "t1",
                "t2",
                "t3",
                "t4",
                "t5",
            ],
            values=[
                1,
                2,
                1,
                10,
                20,
            ],
            min_segment_size=3,
        )
