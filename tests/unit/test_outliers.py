import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_outlier_diagnostic_detects_clear_upper_outlier():
    result = DataAnalysisEngine().analyze(
        "outlier_diagnostic",
        values=[
            10,
            11,
            12,
            13,
            100,
        ],
    ).result

    assert result[
        "outlier_count"
    ] == 1

    assert result[
        "upper_outlier_count"
    ] == 1

    assert result[
        "lower_outlier_count"
    ] == 0

    assert result[
        "outlier_positions"
    ] == [4]

    assert result[
        "outlier_values"
    ] == [100.0]

    assert (
        result[
            "has_outliers"
        ]
        is True
    )


def test_outlier_diagnostic_reports_clean_column():
    result = DataAnalysisEngine().analyze(
        "outlier_diagnostic",
        values=[
            10,
            11,
            12,
            13,
            14,
        ],
    ).result

    assert result[
        "outlier_count"
    ] == 0

    assert (
        result[
            "has_outliers"
        ]
        is False
    )

    assert result[
        "risk_level"
    ] == "none"


def test_outlier_diagnostic_preserves_original_positions_with_missing_values():
    result = DataAnalysisEngine().analyze(
        "outlier_diagnostic",
        values=[
            10,
            None,
            11,
            12,
            100,
        ],
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "non_missing_count"
    ] == 4

    assert result[
        "missing_count"
    ] == 1

    assert result[
        "outlier_positions"
    ] == [4]


def test_outlier_diagnostic_supports_custom_iqr_multiplier():
    values = [
        10,
        11,
        12,
        13,
        100,
    ]

    normal = DataAnalysisEngine().analyze(
        "outlier_diagnostic",
        values=values,
        iqr_multiplier=1.5,
    ).result

    wide = DataAnalysisEngine().analyze(
        "outlier_diagnostic",
        values=values,
        iqr_multiplier=100.0,
    ).result

    assert normal[
        "outlier_count"
    ] == 1

    assert wide[
        "outlier_count"
    ] == 0


def test_outlier_diagnostic_reports_all_missing_column():
    result = DataAnalysisEngine().analyze(
        "outlier_diagnostic",
        values=[
            None,
            np.nan,
            None,
        ],
    ).result

    assert (
        result[
            "is_all_missing"
        ]
        is True
    )

    assert result[
        "non_missing_count"
    ] == 0

    assert result[
        "outlier_count"
    ] == 0

    assert result[
        "risk_level"
    ] == "critical"


def test_outlier_diagnostic_rejects_infinite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "outlier_diagnostic",
            values=[
                1.0,
                2.0,
                np.inf,
                3.0,
            ],
        )


def test_outlier_diagnostic_rejects_invalid_multiplier():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "outlier_diagnostic",
            values=[
                1,
                2,
                3,
            ],
            iqr_multiplier=0.0,
        )
