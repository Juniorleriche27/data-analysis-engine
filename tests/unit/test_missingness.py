import numpy as np
import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_missing_diagnostic_counts_missing_values():
    result = DataAnalysisEngine().analyze(
        "missing_value_diagnostic",
        values=[
            10,
            None,
            30,
            np.nan,
            50,
        ],
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "missing_count"
    ] == 2

    assert result[
        "complete_count"
    ] == 3

    assert result[
        "missing_rate"
    ] == pytest.approx(
        0.4
    )

    assert result[
        "missing_percentage"
    ] == pytest.approx(
        40.0
    )


def test_missing_diagnostic_complete_column():
    result = DataAnalysisEngine().analyze(
        "missing_value_diagnostic",
        values=[
            10,
            20,
            30,
            40,
        ],
    ).result

    assert result[
        "missing_count"
    ] == 0

    assert (
        result[
            "has_missing"
        ]
        is False
    )

    assert (
        result[
            "complete"
        ]
        is True
    )

    assert result[
        "severity"
    ] == "none"

    assert result[
        "first_missing_index"
    ] is None

    assert result[
        "last_missing_index"
    ] is None


def test_missing_diagnostic_all_missing():
    result = DataAnalysisEngine().analyze(
        "missing_value_diagnostic",
        values=[
            None,
            np.nan,
            pd.NA,
        ],
    ).result

    assert result[
        "missing_count"
    ] == 3

    assert result[
        "missing_rate"
    ] == pytest.approx(
        1.0
    )

    assert (
        result[
            "all_missing"
        ]
        is True
    )

    assert result[
        "severity"
    ] == "complete"


def test_missing_diagnostic_detects_missing_runs():
    result = DataAnalysisEngine().analyze(
        "missing_value_diagnostic",
        values=[
            1,
            None,
            None,
            4,
            None,
            6,
            None,
            None,
            None,
        ],
    ).result

    assert result[
        "missing_run_count"
    ] == 3

    assert result[
        "longest_missing_run"
    ] == 3

    assert result[
        "first_missing_index"
    ] == 1

    assert result[
        "last_missing_index"
    ] == 8


def test_missing_diagnostic_recognizes_pandas_missing_values():
    result = DataAnalysisEngine().analyze(
        "missing_value_diagnostic",
        values=[
            pd.Timestamp(
                "2026-01-01"
            ),
            pd.NaT,
            pd.Timestamp(
                "2026-01-03"
            ),
            pd.NA,
        ],
    ).result

    assert result[
        "missing_count"
    ] == 2

    assert result[
        "missing_rate"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "severity"
    ] == "critical"


def test_missing_diagnostic_rejects_empty_input():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "missing_value_diagnostic",
            values=[],
        )
