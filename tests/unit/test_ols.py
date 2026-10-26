import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_ols_fits_exact_simple_linear_relationship():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "y": [
                5.0,
                8.0,
                11.0,
                14.0,
                17.0,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "intercept"
    ] == pytest.approx(
        2.0
    )

    assert result[
        "coefficients"
    ]["x"] == pytest.approx(
        3.0
    )

    assert result[
        "r_squared"
    ] == pytest.approx(
        1.0
    )


def test_ols_fits_multiple_regression():
    x1 = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
    )

    x2 = np.array(
        [
            2.0,
            1.0,
            4.0,
            3.0,
            6.0,
            5.0,
        ]
    )

    y = (
        4.0
        + 2.0 * x1
        - 3.0 * x2
    )

    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares",
        data={
            "x1": x1,
            "x2": x2,
            "y": y,
        },
        target_column="y",
        predictor_columns=[
            "x1",
            "x2",
        ],
    ).result

    assert result[
        "intercept"
    ] == pytest.approx(
        4.0
    )

    assert result[
        "coefficients"
    ]["x1"] == pytest.approx(
        2.0
    )

    assert result[
        "coefficients"
    ]["x2"] == pytest.approx(
        -3.0
    )

    assert result[
        "r_squared"
    ] == pytest.approx(
        1.0
    )


def test_ols_without_intercept():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "y": [
                2.0,
                4.0,
                6.0,
                8.0,
            ],
        },
        target_column="y",
        fit_intercept=False,
    ).result

    assert result[
        "intercept"
    ] is None

    assert result[
        "coefficients"
    ]["x"] == pytest.approx(
        2.0
    )


def test_ols_uses_complete_cases():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares",
        data={
            "x": [
                1.0,
                2.0,
                None,
                4.0,
                5.0,
            ],
            "y": [
                3.0,
                5.0,
                7.0,
                None,
                11.0,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "sample_size"
    ] == 3

    assert result[
        "dropped_count"
    ] == 2

    assert result[
        "complete_case_rate"
    ] == pytest.approx(
        0.6
    )


def test_ols_returns_predictions_and_residuals():
    result = DataAnalysisEngine().analyze(
        "ordinary_least_squares",
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
                4.1,
                5.9,
                8.2,
                9.8,
            ],
        },
        target_column="y",
    ).result

    assert len(
        result[
            "predictions"
        ]
    ) == 5

    assert len(
        result[
            "residuals"
        ]
    ) == 5

    assert result[
        "root_mean_squared_error"
    ] >= 0.0

    assert result[
        "mean_absolute_error"
    ] >= 0.0


def test_ols_rejects_rank_deficient_design():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ordinary_least_squares",
            data={
                "x1": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ],
                "x2": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                ],
                "y": [
                    3.0,
                    5.0,
                    7.0,
                    9.0,
                    11.0,
                ],
            },
            target_column="y",
        )


def test_ols_rejects_non_finite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ordinary_least_squares",
            data={
                "x": [
                    1.0,
                    2.0,
                    np.inf,
                    4.0,
                ],
                "y": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                ],
            },
            target_column="y",
        )


def test_ols_requires_target_column():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ordinary_least_squares",
            data={
                "x": [
                    1.0,
                    2.0,
                    3.0,
                ],
                "y": [
                    2.0,
                    4.0,
                    6.0,
                ],
            },
        )


def test_ols_rejects_insufficient_complete_observations():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ordinary_least_squares",
            data={
                "x1": [
                    1.0,
                    2.0,
                    3.0,
                ],
                "x2": [
                    3.0,
                    1.0,
                    2.0,
                ],
                "y": [
                    2.0,
                    4.0,
                    6.0,
                ],
            },
            target_column="y",
        )
