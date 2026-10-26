import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_non_finite_counts_all_categories():
    result = DataAnalysisEngine().analyze(
        "non_finite_value_diagnostic",
        values=[
            10.0,
            np.inf,
            20.0,
            -np.inf,
            np.nan,
        ],
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "finite_count"
    ] == 2

    assert result[
        "missing_count"
    ] == 1

    assert result[
        "positive_infinity_count"
    ] == 1

    assert result[
        "negative_infinity_count"
    ] == 1

    assert result[
        "infinity_count"
    ] == 2

    assert result[
        "non_finite_count"
    ] == 3


def test_non_finite_reports_positions():
    result = DataAnalysisEngine().analyze(
        "non_finite_value_diagnostic",
        values=[
            np.inf,
            10.0,
            None,
            -np.inf,
        ],
    ).result

    assert result[
        "positive_infinity_positions"
    ] == [
        0,
    ]

    assert result[
        "missing_positions"
    ] == [
        2,
    ]

    assert result[
        "negative_infinity_positions"
    ] == [
        3,
    ]


def test_non_finite_clean_numeric_column():
    result = DataAnalysisEngine().analyze(
        "non_finite_value_diagnostic",
        values=[
            10,
            20,
            30,
        ],
    ).result

    assert result[
        "finite_count"
    ] == 3

    assert result[
        "non_finite_count"
    ] == 0

    assert (
        result[
            "has_non_finite"
        ]
        is False
    )

    assert (
        result[
            "has_infinity"
        ]
        is False
    )

    assert result[
        "risk_level"
    ] == "none"


def test_non_finite_rates_are_correct():
    result = DataAnalysisEngine().analyze(
        "non_finite_value_diagnostic",
        values=[
            1.0,
            2.0,
            np.inf,
            None,
        ],
    ).result

    assert result[
        "finite_rate"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "missing_rate"
    ] == pytest.approx(
        0.25
    )

    assert result[
        "infinity_rate"
    ] == pytest.approx(
        0.25
    )

    assert result[
        "non_finite_rate"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "risk_level"
    ] == "critical"


def test_non_finite_detects_all_non_finite():
    result = DataAnalysisEngine().analyze(
        "non_finite_value_diagnostic",
        values=[
            np.inf,
            -np.inf,
            None,
        ],
    ).result

    assert (
        result[
            "all_non_finite"
        ]
        is True
    )

    assert result[
        "finite_count"
    ] == 0

    assert result[
        "risk_level"
    ] == "critical"


def test_non_finite_rejects_non_numeric_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "non_finite_value_diagnostic",
            values=[
                10,
                "invalid",
                20,
            ],
        )
