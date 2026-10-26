import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_ordinal_encoding_is_first_seen_deterministic():
    result = DataAnalysisEngine().analyze(
        "categorical_encoding",
        values=[
            "B",
            "A",
            "B",
            "C",
        ],
        strategy="ordinal",
    ).result

    assert result[
        "categories"
    ] == [
        "B",
        "A",
        "C",
    ]

    assert result[
        "mapping"
    ] == {
        "B": 0,
        "A": 1,
        "C": 2,
    }

    assert result[
        "values"
    ] == [
        0,
        1,
        0,
        2,
    ]


def test_ordinal_encoding_represents_missing_values():
    result = DataAnalysisEngine().analyze(
        "categorical_encoding",
        values=[
            "A",
            None,
            "B",
            pd.NA,
        ],
        strategy="ordinal",
    ).result

    assert result[
        "missing_count"
    ] == 2

    assert result[
        "missing_code"
    ] == 2

    assert result[
        "values"
    ] == [
        0,
        2,
        1,
        2,
    ]


def test_one_hot_encoding():
    result = DataAnalysisEngine().analyze(
        "categorical_encoding",
        values=[
            "A",
            "B",
            "A",
        ],
        strategy="one_hot",
    ).result

    columns = result[
        "encoded_columns"
    ]

    assert columns[
        "A"
    ] == [
        1,
        0,
        1,
    ]

    assert columns[
        "B"
    ] == [
        0,
        1,
        0,
    ]

    assert result[
        "output_column_count"
    ] == 2


def test_one_hot_encoding_has_missing_indicator():
    result = DataAnalysisEngine().analyze(
        "categorical_encoding",
        values=[
            "A",
            None,
            "B",
        ],
        strategy="one_hot",
    ).result

    assert result[
        "encoded_columns"
    ][
        "__MISSING__"
    ] == [
        0,
        1,
        0,
    ]

    assert result[
        "output_column_count"
    ] == 3


def test_encoding_supports_custom_missing_label():
    result = DataAnalysisEngine().analyze(
        "categorical_encoding",
        values=[
            "A",
            None,
            "B",
        ],
        strategy="one_hot",
        missing_label="MISSING",
    ).result

    assert (
        "MISSING"
        in result[
            "encoded_columns"
        ]
    )


def test_encoding_rejects_invalid_strategy():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "categorical_encoding",
            values=[
                "A",
                "B",
            ],
            strategy="unknown",
        )


def test_one_hot_rejects_missing_label_collision():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "categorical_encoding",
            values=[
                "__MISSING__",
                None,
                "A",
            ],
            strategy="one_hot",
        )
