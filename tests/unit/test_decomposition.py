import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_centered_moving_average_with_odd_window():
    result = DataAnalysisEngine().analyze(
        "moving_average_trend",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
            "t7",
        ],
        values=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ],
        window=3,
    )

    assert result.result[
        "trend"
    ] == [
        None,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        None,
    ]


def test_centered_moving_average_with_even_window():
    result = DataAnalysisEngine().analyze(
        "moving_average_trend",
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
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ],
        window=4,
    )

    assert result.result[
        "trend"
    ] == [
        None,
        None,
        3.0,
        4.0,
        5.0,
        6.0,
        None,
        None,
    ]


def test_moving_average_rejects_invalid_window():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "moving_average_trend",
            time=[
                "t1",
                "t2",
                "t3",
            ],
            values=[
                1,
                2,
                3,
            ],
            window=1,
        )


def test_additive_decomposition_recovers_seasonal_pattern():
    trend = list(
        range(
            10,
            22,
        )
    )

    seasonal = [
        2,
        -1,
        -2,
        1,
    ] * 3

    values = [
        trend_value + seasonal_value
        for trend_value, seasonal_value
        in zip(
            trend,
            seasonal,
        )
    ]

    result = DataAnalysisEngine().analyze(
        "additive_seasonal_decomposition",
        time=[
            f"t{index}"
            for index in range(
                1,
                13,
            )
        ],
        values=values,
        period=4,
    )

    assert result.result[
        "seasonal_pattern"
    ] == pytest.approx(
        [
            2,
            -1,
            -2,
            1,
        ]
    )


def test_additive_decomposition_residual_is_near_zero():
    trend = list(
        range(
            10,
            22,
        )
    )

    seasonal = [
        2,
        -1,
        -2,
        1,
    ] * 3

    values = [
        trend_value + seasonal_value
        for trend_value, seasonal_value
        in zip(
            trend,
            seasonal,
        )
    ]

    result = DataAnalysisEngine().analyze(
        "additive_seasonal_decomposition",
        time=[
            f"t{index}"
            for index in range(
                1,
                13,
            )
        ],
        values=values,
        period=4,
    )

    residuals = [
        value
        for value in result.result[
            "residual"
        ]
        if value is not None
    ]

    assert residuals == pytest.approx(
        [0.0] * len(residuals)
    )


def test_additive_decomposition_requires_two_periods():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "additive_seasonal_decomposition",
            time=[
                "t1",
                "t2",
                "t3",
                "t4",
                "t5",
            ],
            values=[
                10,
                20,
                10,
                20,
                10,
            ],
            period=3,
        )
