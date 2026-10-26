import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_variance_diagnostic_detects_zero_variance():
    result = DataAnalysisEngine().analyze(
        "variance_diagnostic",
        values=[
            5,
            5,
            5,
            5,
        ],
    ).result

    assert result["variance"] == pytest.approx(0.0)
    assert result["is_zero_variance"] is True
    assert result["is_near_zero_variance"] is False
    assert result["variance_status"] == "zero_variance"
    assert result["risk_level"] == "critical"


def test_variance_diagnostic_detects_near_zero_variance():
    values = [
        1_000_000.0,
        1_000_000.1,
        999_999.9,
        1_000_000.05,
    ]

    result = DataAnalysisEngine().analyze(
        "variance_diagnostic",
        values=values,
        relative_variance_threshold=1e-6,
    ).result

    assert result["variance"] > 0.0
    assert result["relative_variance"] < 1e-6
    assert result["is_zero_variance"] is False
    assert result["is_near_zero_variance"] is True
    assert result["variance_status"] == "near_zero_variance"


def test_variance_diagnostic_reports_adequate_variance():
    result = DataAnalysisEngine().analyze(
        "variance_diagnostic",
        values=[
            -10,
            -5,
            0,
            5,
            10,
        ],
    ).result

    assert result["variance"] > 0.0
    assert result["is_zero_variance"] is False
    assert result["is_near_zero_variance"] is False
    assert result["variance_status"] == "adequate_variance"
    assert result["risk_level"] == "none"


def test_variance_diagnostic_preserves_missing_information():
    result = DataAnalysisEngine().analyze(
        "variance_diagnostic",
        values=[
            1.0,
            None,
            2.0,
            np.nan,
            3.0,
        ],
    ).result

    assert result["total_count"] == 5
    assert result["non_missing_count"] == 3
    assert result["missing_count"] == 2
    assert result["variance"] == pytest.approx(
        2 / 3
    )


def test_variance_diagnostic_reports_all_missing():
    result = DataAnalysisEngine().analyze(
        "variance_diagnostic",
        values=[
            None,
            np.nan,
            None,
        ],
    ).result

    assert result["is_all_missing"] is True
    assert result["variance"] is None
    assert result["risk_level"] == "critical"


def test_variance_diagnostic_rejects_infinite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "variance_diagnostic",
            values=[
                1.0,
                np.inf,
                2.0,
            ],
        )


def test_variance_diagnostic_rejects_invalid_threshold():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "variance_diagnostic",
            values=[
                1,
                2,
                3,
            ],
            relative_variance_threshold=-1.0,
        )
