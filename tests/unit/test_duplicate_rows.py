import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_duplicate_rows_are_detected():
    result = DataAnalysisEngine().analyze(
        "duplicate_row_diagnostic",
        data={
            "category": [
                "A",
                "A",
                "B",
                "C",
            ],
            "value": [
                10,
                10,
                20,
                30,
            ],
        },
    ).result

    assert result[
        "total_count"
    ] == 4

    assert result[
        "unique_row_count"
    ] == 3

    assert result[
        "duplicate_row_count"
    ] == 2

    assert result[
        "excess_duplicate_count"
    ] == 1

    assert result[
        "duplicate_group_count"
    ] == 1

    assert result[
        "duplicate_positions"
    ] == [
        0,
        1,
    ]

    assert result[
        "excess_duplicate_positions"
    ] == [
        1,
    ]


def test_duplicate_rows_preserve_missing_values():
    result = DataAnalysisEngine().analyze(
        "duplicate_row_diagnostic",
        data={
            "category": [
                "A",
                "A",
                "B",
            ],
            "value": [
                None,
                None,
                10,
            ],
        },
    ).result

    assert result[
        "total_count"
    ] == 3

    assert result[
        "missing_cell_count"
    ] == 2

    assert result[
        "duplicate_row_count"
    ] == 2

    assert result[
        "excess_duplicate_count"
    ] == 1


def test_duplicate_rows_reports_clean_dataset():
    result = DataAnalysisEngine().analyze(
        "duplicate_row_diagnostic",
        data={
            "x": [
                1,
                2,
                3,
            ],
            "y": [
                10,
                20,
                30,
            ],
        },
    ).result

    assert (
        result[
            "has_duplicates"
        ]
        is False
    )

    assert result[
        "duplicate_row_count"
    ] == 0

    assert result[
        "excess_duplicate_count"
    ] == 0

    assert result[
        "risk_level"
    ] == "none"


def test_duplicate_rows_detects_all_identical_rows():
    result = DataAnalysisEngine().analyze(
        "duplicate_row_diagnostic",
        data={
            "x": [
                1,
                1,
                1,
            ],
            "y": [
                "A",
                "A",
                "A",
            ],
        },
    ).result

    assert (
        result[
            "all_rows_identical"
        ]
        is True
    )

    assert result[
        "unique_row_count"
    ] == 1

    assert result[
        "excess_duplicate_count"
    ] == 2

    assert result[
        "risk_level"
    ] == "critical"


def test_duplicate_rows_calculates_rates():
    result = DataAnalysisEngine().analyze(
        "duplicate_row_diagnostic",
        data={
            "x": [
                1,
                1,
                2,
                3,
                4,
            ]
        },
    ).result

    assert result[
        "duplicate_row_rate"
    ] == pytest.approx(
        0.4
    )

    assert result[
        "excess_duplicate_rate"
    ] == pytest.approx(
        0.2
    )

    assert result[
        "risk_level"
    ] == "critical"


def test_duplicate_rows_rejects_empty_data():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "duplicate_row_diagnostic",
            data={},
        )
