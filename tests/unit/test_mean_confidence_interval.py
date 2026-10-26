import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_mean_confidence_interval_contains_sample_mean():
    result = DataAnalysisEngine().analyze(
        "mean_confidence_interval",
        values=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        coverage=0.95,
    ).result

    assert result["mean"] == pytest.approx(3.0)

    assert (
        result["lower_bound"]
        < result["mean"]
        < result["upper_bound"]
    )

    assert result["coverage"] == pytest.approx(0.95)


def test_mean_confidence_interval_margin_identity():
    result = DataAnalysisEngine().analyze(
        "mean_confidence_interval",
        values=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
    ).result

    assert result[
        "margin_of_error"
    ] == pytest.approx(
        result["critical_value"]
        * result["standard_error"]
    )

    assert result[
        "interval_width"
    ] == pytest.approx(
        2.0
        * result["margin_of_error"]
    )


def test_mean_confidence_interval_constant_series():
    result = DataAnalysisEngine().analyze(
        "mean_confidence_interval",
        values=[
            10.0,
            10.0,
            10.0,
            10.0,
        ],
    ).result

    assert result["is_constant"] is True
    assert result["standard_error"] == pytest.approx(0.0)
    assert result["lower_bound"] == pytest.approx(10.0)
    assert result["upper_bound"] == pytest.approx(10.0)


def test_mean_confidence_interval_preserves_missing_information():
    result = DataAnalysisEngine().analyze(
        "mean_confidence_interval",
        values=[
            1.0,
            None,
            2.0,
            np.nan,
            3.0,
        ],
    ).result

    assert result["total_count"] == 5
    assert result["sample_size"] == 3
    assert result["missing_count"] == 2


def test_mean_confidence_interval_rejects_single_observation():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "mean_confidence_interval",
            values=[
                1.0,
            ],
        )
