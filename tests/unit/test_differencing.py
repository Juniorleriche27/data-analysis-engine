import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_first_difference():
    result = DataAnalysisEngine().analyze(
        "first_difference",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
        ],
        values=[
            100,
            105,
            103,
            110,
        ],
    )

    assert result.result[
        "differences"
    ] == pytest.approx(
        [
            5,
            -2,
            7,
        ]
    )

    assert (
        result.result[
            "difference_count"
        ]
        == 3
    )


def test_first_difference_timestamps_align():
    result = DataAnalysisEngine().analyze(
        "first_difference",
        time=[
            "t1",
            "t2",
            "t3",
        ],
        values=[
            10,
            15,
            21,
        ],
    )

    assert result.result[
        "timestamps"
    ] == [
        "t2",
        "t3",
    ]


def test_seasonal_difference():
    result = DataAnalysisEngine().analyze(
        "seasonal_difference",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
        ],
        values=[
            10,
            20,
            30,
            15,
            25,
            35,
        ],
        period=3,
    )

    assert result.result[
        "differences"
    ] == pytest.approx(
        [
            5,
            5,
            5,
        ]
    )


def test_seasonal_difference_rejects_invalid_period():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "seasonal_difference",
            time=[
                "t1",
                "t2",
                "t3",
            ],
            values=[
                10,
                20,
                30,
            ],
            period=3,
        )
