import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


# ============================================================
# NEWEY-WEST
# ============================================================

def test_newey_west_returns_hac_standard_errors():
    result = DataAnalysisEngine().analyze(
        "newey_west_ols_inference",
        data={
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
                3.1,
                5.4,
                7.0,
                9.5,
                11.0,
                13.6,
                15.1,
                17.4,
            ],
        },
        target_column="y",
        max_lag=2,
    ).result

    assert result[
        "covariance_type"
    ] == "newey_west_hac"

    assert result[
        "max_lag"
    ] == 2

    assert result[
        "coefficients"
    ]["x"][
        "standard_error"
    ] >= 0.0


def test_newey_west_auto_lag_is_valid():
    result = DataAnalysisEngine().analyze(
        "newey_west_ols_inference",
        data={
            "x": list(
                range(
                    1,
                    31,
                )
            ),
            "y": [
                2.1 * value
                + (
                    0.2
                    if value % 2
                    else -0.2
                )
                for value
                in range(
                    1,
                    31,
                )
            ],
        },
        target_column="y",
    ).result

    assert (
        0
        <= result[
            "max_lag"
        ]
        < result[
            "sample_size"
        ]
    )


def test_newey_west_confidence_interval_contains_estimate():
    result = DataAnalysisEngine().analyze(
        "newey_west_ols_inference",
        data={
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
                7.2,
                9.1,
                11.3,
                13.0,
                15.2,
                17.1,
            ],
        },
        target_column="y",
        max_lag=1,
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
        <= slope[
            "coefficient"
        ]
        <= interval[
            "upper_bound"
        ]
    )


def test_newey_west_tracks_missing_rows():
    result = DataAnalysisEngine().analyze(
        "newey_west_ols_inference",
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
        max_lag=1,
    ).result

    assert result[
        "sample_size"
    ] == 4

    assert result[
        "dropped_count"
    ] == 2


def test_newey_west_rejects_invalid_lag():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "newey_west_ols_inference",
            data={
                "x": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                "y": [
                    2.0,
                    4.1,
                    6.0,
                    8.2,
                ],
            },
            target_column="y",
            max_lag=4,
        )


# ============================================================
# NESTED F TEST
# ============================================================

def test_nested_f_test_detects_useful_added_predictor():
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
            9.0,
            10.0,
        ]
    )

    x2 = np.array(
        [
            3.0,
            1.0,
            4.0,
            2.0,
            6.0,
            5.0,
            9.0,
            7.0,
            10.0,
            8.0,
        ]
    )

    y = (
        2.0
        + 1.5 * x1
        + 4.0 * x2
    )

    result = DataAnalysisEngine().analyze(
        "nested_ols_f_test",
        data={
            "x1": x1,
            "x2": x2,
            "y": y,
        },
        target_column="y",
        restricted_predictors=[
            "x1",
        ],
        full_predictors=[
            "x1",
            "x2",
        ],
    ).result

    assert result[
        "rss_full"
    ] < result[
        "rss_restricted"
    ]

    assert result[
        "p_value"
    ] < 0.05

    assert result[
        "added_predictors"
    ] == [
        "x2",
    ]


def test_nested_f_test_supports_intercept_only_restricted_model():
    x = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    y = (
        3.0
        + 2.0 * x
    )

    result = DataAnalysisEngine().analyze(
        "nested_ols_f_test",
        data={
            "x": x,
            "y": y,
        },
        target_column="y",
        restricted_predictors=[],
        full_predictors=[
            "x",
        ],
    ).result

    assert result[
        "restriction_count"
    ] == 1

    assert result[
        "p_value"
    ] < 0.05


def test_nested_f_test_uses_same_complete_sample():
    result = DataAnalysisEngine().analyze(
        "nested_ols_f_test",
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
                None,
                1.0,
                5.0,
                3.0,
                7.0,
            ],
            "y": [
                4.0,
                5.0,
                7.0,
                9.0,
                10.0,
                14.0,
            ],
        },
        target_column="y",
        restricted_predictors=[
            "x1",
        ],
        full_predictors=[
            "x1",
            "x2",
        ],
    ).result

    assert result[
        "sample_size"
    ] == 5

    assert result[
        "dropped_count"
    ] == 1


def test_nested_f_test_rejects_non_nested_models():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "nested_ols_f_test",
            data={
                "x1": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ],
                "x2": [
                    5.0,
                    1.0,
                    4.0,
                    2.0,
                    3.0,
                ],
                "y": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                ],
            },
            target_column="y",
            restricted_predictors=[
                "x2",
            ],
            full_predictors=[
                "x1",
            ],
        )


# ============================================================
# JARQUE-BERA
# ============================================================

def test_jarque_bera_returns_valid_probability():
    result = DataAnalysisEngine().analyze(
        "jarque_bera_residual_normality_test",
        data={
            "x": list(
                range(
                    1,
                    13,
                )
            ),
            "y": [
                2.1,
                4.2,
                5.8,
                8.3,
                9.9,
                12.1,
                14.2,
                15.8,
                18.1,
                20.2,
                21.9,
                24.1,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "jarque_bera_statistic"
    ] >= 0.0

    assert (
        0.0
        <= result[
            "p_value"
        ]
        <= 1.0
    )


def test_jarque_bera_detects_extreme_residual_shape():
    x = np.arange(
        1,
        31,
        dtype=float,
    )

    y = (
        2.0
        * x
    )

    y[
        -1
    ] += 100.0

    result = DataAnalysisEngine().analyze(
        "jarque_bera_residual_normality_test",
        data={
            "x": x,
            "y": y,
        },
        target_column="y",
    ).result

    assert result[
        "p_value"
    ] < 0.05


def test_jarque_bera_handles_perfect_fit():
    result = DataAnalysisEngine().analyze(
        "jarque_bera_residual_normality_test",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "y": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "perfect_fit"
    ] is True

    assert result[
        "jarque_bera_statistic"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )


def test_jarque_bera_tracks_dropped_rows():
    result = DataAnalysisEngine().analyze(
        "jarque_bera_residual_normality_test",
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
                2.1,
                4.2,
                6.0,
                None,
                10.1,
                12.3,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "sample_size"
    ] == 4

    assert result[
        "dropped_count"
    ] == 2
