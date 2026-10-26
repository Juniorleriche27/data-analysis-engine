import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_DATA = {
    "x": [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ],
    "y": [
        3.2,
        5.1,
        7.4,
        8.7,
        11.2,
        12.8,
        15.3,
        17.1,
    ],
}


def test_ols_inference_returns_coefficient_statistics():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
    ).result

    assert "intercept" in result[
        "coefficients"
    ]

    assert "x" in result[
        "coefficients"
    ]

    slope = result[
        "coefficients"
    ]["x"]

    assert slope[
        "standard_error"
    ] > 0.0

    assert 0.0 <= slope[
        "p_value"
    ] <= 1.0


def test_ols_inference_slope_is_near_expected_value():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
    ).result

    slope = result[
        "coefficients"
    ]["x"]

    assert slope[
        "coefficient"
    ] == pytest.approx(
        2.0,
        abs=0.15,
    )

    assert slope[
        "p_value"
    ] < 0.001


def test_ols_inference_confidence_interval_contains_slope():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
        confidence_level=0.95,
    ).result

    slope = result[
        "coefficients"
    ]["x"]

    interval = slope[
        "confidence_interval"
    ]

    assert (
        interval[
            "lower_bound"
        ]
        < slope[
            "coefficient"
        ]
        < interval[
            "upper_bound"
        ]
    )


def test_ols_inference_supports_hc0():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
        covariance_type="hc0",
    ).result

    assert result[
        "covariance_type"
    ] == "hc0"

    assert result[
        "coefficients"
    ]["x"][
        "standard_error"
    ] >= 0.0


def test_ols_inference_supports_hc1():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
        covariance_type="hc1",
    ).result

    assert result[
        "covariance_type"
    ] == "hc1"

    assert result[
        "coefficients"
    ]["x"][
        "standard_error"
    ] >= 0.0


def test_hc1_standard_errors_are_not_smaller_than_hc0():
    engine = DataAnalysisEngine()

    hc0 = engine.analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
        covariance_type="hc0",
    ).result

    hc1 = engine.analyze(
        "ordinary_least_squares_inference",
        data=_DATA,
        target_column="y",
        covariance_type="hc1",
    ).result

    assert hc1[
        "coefficients"
    ]["x"][
        "standard_error"
    ] >= hc0[
        "coefficients"
    ]["x"][
        "standard_error"
    ]


def test_ols_inference_multiple_regression():
    x1 = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ]
    )

    x2 = np.array(
        [
            2.0,
            5.0,
            1.0,
            6.0,
            3.0,
            8.0,
            4.0,
            7.0,
        ]
    )

    noise = np.array(
        [
            0.1,
            -0.2,
            0.3,
            -0.1,
            0.2,
            -0.3,
            0.1,
            -0.1,
        ]
    )

    y = (
        4.0
        + 2.0 * x1
        - 1.5 * x2
        + noise
    )

    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data={
            "x1": x1,
            "x2": x2,
            "y": y,
        },
        target_column="y",
    ).result

    assert result[
        "coefficients"
    ]["x1"][
        "coefficient"
    ] == pytest.approx(
        2.0,
        abs=0.2,
    )

    assert result[
        "coefficients"
    ]["x2"][
        "coefficient"
    ] == pytest.approx(
        -1.5,
        abs=0.2,
    )


def test_ols_inference_uses_complete_cases():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares_inference",
        data={
            "x": [
                1.0,
                2.0,
                None,
                4.0,
                5.0,
                6.0,
            ],
            "y": [
                3.0,
                5.0,
                7.0,
                None,
                11.0,
                13.0,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "total_count"
    ] == 6

    assert result[
        "sample_size"
    ] == 4

    assert result[
        "dropped_count"
    ] == 2


def test_ols_inference_rejects_invalid_covariance_type():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ordinary_least_squares_inference",
            data=_DATA,
            target_column="y",
            covariance_type="invalid",
        )


def test_ols_inference_rejects_rank_deficiency():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ordinary_least_squares_inference",
            data={
                "x1": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ],
                "x2": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                    12.0,
                ],
                "y": [
                    3.0,
                    5.0,
                    7.0,
                    9.0,
                    11.0,
                    13.0,
                ],
            },
            target_column="y",
        )
