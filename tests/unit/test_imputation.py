import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_mean_imputation():
    result = DataAnalysisEngine().analyze(
        "missing_value_imputation",
        values=[
            1.0,
            None,
            3.0,
        ],
        strategy="mean",
    ).result

    assert result[
        "fill_value"
    ] == pytest.approx(
        2.0
    )

    assert result[
        "values"
    ] == [
        1.0,
        2.0,
        3.0,
    ]

    assert result[
        "imputed_positions"
    ] == [1]


def test_median_imputation():
    result = DataAnalysisEngine().analyze(
        "missing_value_imputation",
        values=[
            1.0,
            None,
            100.0,
            3.0,
        ],
        strategy="median",
    ).result

    assert result[
        "fill_value"
    ] == pytest.approx(
        3.0
    )

    assert result[
        "missing_count_after"
    ] == 0


def test_mode_imputation_is_deterministic():
    result = DataAnalysisEngine().analyze(
        "missing_value_imputation",
        values=[
            "B",
            "A",
            "B",
            "A",
            None,
        ],
        strategy="mode",
    ).result

    assert result[
        "fill_value"
    ] == "B"

    assert result[
        "values"
    ][4] == "B"


def test_constant_imputation():
    result = DataAnalysisEngine().analyze(
        "missing_value_imputation",
        values=[
            "A",
            None,
            "B",
        ],
        strategy="constant",
        fill_value="UNKNOWN",
    ).result

    assert result[
        "values"
    ] == [
        "A",
        "UNKNOWN",
        "B",
    ]

    assert result[
        "imputed_count"
    ] == 1


def test_constant_imputation_handles_all_missing_column():
    result = DataAnalysisEngine().analyze(
        "missing_value_imputation",
        values=[
            None,
            pd.NA,
            None,
        ],
        strategy="constant",
        fill_value=0,
    ).result

    assert result[
        "values"
    ] == [
        0,
        0,
        0,
    ]

    assert result[
        "imputed_count"
    ] == 3


def test_imputation_noop_when_no_missing_values():
    result = DataAnalysisEngine().analyze(
        "missing_value_imputation",
        values=[
            1,
            2,
            3,
        ],
        strategy="median",
    ).result

    assert result[
        "changed"
    ] is False

    assert result[
        "imputed_count"
    ] == 0

    assert result[
        "values"
    ] == [
        1,
        2,
        3,
    ]


def test_mean_imputation_rejects_text_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "missing_value_imputation",
            values=[
                "10",
                None,
                "20",
            ],
            strategy="mean",
        )
