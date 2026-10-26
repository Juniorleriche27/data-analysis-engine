import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_multicollinearity_detects_perfect_linear_dependence():
    result = DataAnalysisEngine().analyze(
        "multicollinearity_diagnostic",
        data={
            "x1": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "x2": [
                2,
                4,
                6,
                8,
                10,
                12,
            ],
        },
    ).result

    assert (
        result[
            "has_multicollinearity"
        ]
        is True
    )

    assert (
        result[
            "maximum_vif_infinite"
        ]
        is True
    )

    assert result[
        "risk_level"
    ] == "critical"


def test_multicollinearity_reports_low_dependency():
    result = DataAnalysisEngine().analyze(
        "multicollinearity_diagnostic",
        data={
            "x1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            "x2": [
                4,
                1,
                7,
                3,
                8,
                2,
                6,
                5,
            ],
        },
    ).result

    assert (
        result[
            "maximum_absolute_correlation"
        ]
        < 0.8
    )

    assert (
        result[
            "has_multicollinearity"
        ]
        is False
    )

    assert result[
        "risk_level"
    ] == "none"


def test_multicollinearity_returns_correlation_matrix():
    result = DataAnalysisEngine().analyze(
        "multicollinearity_diagnostic",
        data={
            "a": [
                1,
                2,
                4,
                7,
                11,
                16,
            ],
            "b": [
                3,
                8,
                2,
                10,
                6,
                14,
            ],
        },
    ).result

    matrix = result[
        "correlation_matrix"
    ]

    assert matrix["a"]["a"] == pytest.approx(
        1.0
    )

    assert matrix["b"]["b"] == pytest.approx(
        1.0
    )

    assert matrix["a"]["b"] == pytest.approx(
        matrix["b"]["a"]
    )


def test_multicollinearity_uses_complete_cases():
    result = DataAnalysisEngine().analyze(
        "multicollinearity_diagnostic",
        data={
            "x1": [
                1,
                2,
                None,
                4,
                5,
                6,
            ],
            "x2": [
                6,
                1,
                4,
                3,
                None,
                5,
            ],
        },
    ).result

    assert result[
        "total_count"
    ] == 6

    assert result[
        "complete_row_count"
    ] == 4

    assert result[
        "dropped_row_count"
    ] == 2


def test_multicollinearity_rejects_constant_column():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "multicollinearity_diagnostic",
            data={
                "x1": [
                    1,
                    1,
                    1,
                    1,
                ],
                "x2": [
                    1,
                    2,
                    3,
                    4,
                ],
            },
        )


def test_multicollinearity_rejects_non_finite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "multicollinearity_diagnostic",
            data={
                "x1": [
                    1,
                    2,
                    3,
                    4,
                ],
                "x2": [
                    1,
                    2,
                    np.inf,
                    4,
                ],
            },
        )


def test_multicollinearity_requires_two_columns():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "multicollinearity_diagnostic",
            data={
                "x1": [
                    1,
                    2,
                    3,
                    4,
                ],
            },
        )
