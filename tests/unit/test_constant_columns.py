import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_constant_column_is_detected():
    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=[
            5,
            5,
            None,
            5,
            5,
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
        "unique_count"
    ] == 1

    assert result[
        "dominant_value"
    ] == 5

    assert result[
        "dominant_ratio"
    ] == pytest.approx(
        1.0
    )

    assert (
        result[
            "is_constant"
        ]
        is True
    )

    assert (
        result[
            "is_quasi_constant"
        ]
        is False
    )

    assert result[
        "risk_level"
    ] == "critical"


def test_quasi_constant_column_is_detected():
    values = (
        ["A"] * 19
        + ["B"]
    )

    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=values,
        dominance_threshold=0.95,
    ).result

    assert result[
        "unique_count"
    ] == 2

    assert result[
        "dominant_value"
    ] == "A"

    assert result[
        "dominant_count"
    ] == 19

    assert result[
        "dominant_ratio"
    ] == pytest.approx(
        0.95
    )

    assert (
        result[
            "is_constant"
        ]
        is False
    )

    assert (
        result[
            "is_quasi_constant"
        ]
        is True
    )

    assert result[
        "risk_level"
    ] == "high"


def test_quasi_constant_threshold_is_configurable():
    values = (
        [1] * 9
        + [2]
    )

    engine = DataAnalysisEngine()

    strict = engine.analyze(
        "constant_column_diagnostic",
        values=values,
        dominance_threshold=0.95,
    ).result

    permissive = engine.analyze(
        "constant_column_diagnostic",
        values=values,
        dominance_threshold=0.90,
    ).result

    assert (
        strict[
            "is_quasi_constant"
        ]
        is False
    )

    assert (
        permissive[
            "is_quasi_constant"
        ]
        is True
    )


def test_balanced_column_is_not_quasi_constant():
    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=[
            "A",
            "A",
            "B",
            "B",
        ],
        dominance_threshold=0.95,
    ).result

    assert result[
        "dominant_ratio"
    ] == pytest.approx(
        0.5
    )

    assert (
        result[
            "is_constant"
        ]
        is False
    )

    assert (
        result[
            "is_quasi_constant"
        ]
        is False
    )

    assert result[
        "risk_level"
    ] == "none"


def test_all_missing_column_is_reported():
    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=[
            None,
            pd.NA,
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
        "unique_count"
    ] == 0

    assert result[
        "dominant_value"
    ] is None

    assert result[
        "dominant_ratio"
    ] is None

    assert result[
        "risk_level"
    ] == "critical"


def test_constant_column_rejects_invalid_threshold():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "constant_column_diagnostic",
            values=[
                1,
                1,
                2,
            ],
            dominance_threshold=0.0,
        )
