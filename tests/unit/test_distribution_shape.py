import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_distribution_shape_symmetric_series_has_zero_skewness():
    result = DataAnalysisEngine().analyze(
        "distribution_shape_statistics",
        values=[
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ],
    ).result

    assert result[
        "skewness"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "adjusted_skewness"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "skew_direction"
    ] == "approximately_symmetric"


def test_distribution_shape_known_excess_kurtosis():
    result = DataAnalysisEngine().analyze(
        "distribution_shape_statistics",
        values=[
            -2.0,
            -1.0,
            0.0,
            1.0,
            2.0,
        ],
    ).result

    assert result[
        "excess_kurtosis"
    ] == pytest.approx(
        -1.3
    )

    assert result[
        "adjusted_excess_kurtosis"
    ] == pytest.approx(
        -1.2
    )

    assert result[
        "kurtosis"
    ] == pytest.approx(
        1.7
    )


def test_distribution_shape_detects_right_skew():
    result = DataAnalysisEngine().analyze(
        "distribution_shape_statistics",
        values=[
            1.0,
            1.0,
            1.0,
            2.0,
            20.0,
        ],
    ).result

    assert result[
        "skewness"
    ] > 0.5

    assert result[
        "skew_direction"
    ] == "right"


def test_distribution_shape_preserves_missing_information():
    result = DataAnalysisEngine().analyze(
        "distribution_shape_statistics",
        values=[
            1.0,
            None,
            2.0,
            np.nan,
            3.0,
        ],
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "sample_size"
    ] == 3

    assert result[
        "missing_count"
    ] == 2


def test_distribution_shape_constant_column_is_reported():
    result = DataAnalysisEngine().analyze(
        "distribution_shape_statistics",
        values=[
            5.0,
            5.0,
            5.0,
            5.0,
        ],
    ).result

    assert result[
        "is_constant"
    ] is True

    assert result[
        "skewness"
    ] is None

    assert result[
        "excess_kurtosis"
    ] is None

    assert result[
        "shape_status"
    ] == "constant"


def test_distribution_shape_small_sample_bias_correction_availability():
    result = DataAnalysisEngine().analyze(
        "distribution_shape_statistics",
        values=[
            1.0,
            2.0,
        ],
    ).result

    assert result[
        "skewness"
    ] is not None

    assert result[
        "adjusted_skewness"
    ] is None

    assert result[
        "adjusted_excess_kurtosis"
    ] is None


def test_distribution_shape_rejects_non_finite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "distribution_shape_statistics",
            values=[
                1.0,
                np.inf,
                3.0,
            ],
        )
