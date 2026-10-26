import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_constant_diagnostic_detects_constant_column():
    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=[
            5,
            5,
            5,
            5,
            5,
        ],
    ).result

    assert result[
        "unique_count"
    ] == 1

    assert (
        result[
            "is_constant"
        ]
        is True
    )

    assert (
        result[
            "is_near_constant"
        ]
        is False
    )

    assert result[
        "variability_status"
    ] == "constant"


def test_constant_diagnostic_detects_near_constant_column():
    values = (
        ["A"] * 19
        + ["B"]
    )

    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=values,
        dominance_threshold=0.95,
    ).result

    assert (
        result[
            "is_constant"
        ]
        is False
    )

    assert (
        result[
            "is_near_constant"
        ]
        is True
    )

    assert result[
        "dominant_value"
    ] == "A"

    assert result[
        "dominant_ratio"
    ] == pytest.approx(
        0.95
    )


def test_constant_diagnostic_detects_variable_column():
    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=[
            1,
            2,
            3,
            4,
            5,
        ],
    ).result

    assert (
        result[
            "is_constant"
        ]
        is False
    )

    assert (
        result[
            "is_near_constant"
        ]
        is False
    )

    assert result[
        "variability_status"
    ] == "variable"

    assert (
        result[
            "informative_variation"
        ]
        is True
    )


def test_constant_diagnostic_ignores_missing_values():
    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=[
            7,
            None,
            7,
            np.nan,
            7,
        ],
    ).result

    assert result[
        "observed_count"
    ] == 3

    assert result[
        "unique_count"
    ] == 1

    assert (
        result[
            "is_constant"
        ]
        is True
    )


def test_constant_diagnostic_supports_custom_threshold():
    values = (
        ["A"] * 8
        + ["B"] * 2
    )

    result = DataAnalysisEngine().analyze(
        "constant_column_diagnostic",
        values=values,
        dominance_threshold=0.80,
    ).result

    assert result[
        "dominant_ratio"
    ] == pytest.approx(
        0.8
    )

    assert (
        result[
            "is_near_constant"
        ]
        is True
    )


def test_constant_diagnostic_rejects_invalid_threshold():
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
            dominance_threshold=1.0,
        )
