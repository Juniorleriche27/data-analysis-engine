import pandas as pd

from data_analysis_engine import DataAnalysisEngine


def test_type_consistency_reports_homogeneous_numeric_column():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
        values=[
            1,
            2,
            3,
            4,
        ],
    ).result

    assert result[
        "type_families"
    ] == ["numeric"]

    assert (
        result[
            "is_homogeneous"
        ]
        is True
    )

    assert (
        result[
            "has_mixed_types"
        ]
        is False
    )

    assert result[
        "risk_level"
    ] == "none"


def test_type_consistency_detects_numeric_strings():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
        values=[
            "10",
            "20.5",
            "-3",
        ],
    ).result

    assert result[
        "numeric_string_count"
    ] == 3

    assert (
        result[
            "has_numeric_strings"
        ]
        is True
    )

    assert result[
        "consistency_status"
    ] == "numeric_stored_as_text"

    assert result[
        "recommended_target_type"
    ] == "numeric"


def test_type_consistency_detects_numeric_coercible_mixture():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
        values=[
            10,
            "20",
            30,
            "40.5",
        ],
    ).result

    assert set(
        result[
            "type_families"
        ]
    ) == {
        "numeric",
        "numeric_string",
    }

    assert (
        result[
            "is_numeric_coercible"
        ]
        is True
    )

    assert (
        result[
            "has_mixed_types"
        ]
        is True
    )

    assert result[
        "consistency_status"
    ] == "numeric_coercible_mixed"


def test_type_consistency_detects_incompatible_mixed_types():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
        values=[
            10,
            "hello",
            20,
            "world",
        ],
    ).result

    assert (
        result[
            "has_mixed_types"
        ]
        is True
    )

    assert (
        result[
            "is_numeric_coercible"
        ]
        is False
    )

    assert result[
        "consistency_status"
    ] == "incompatible_mixed_types"

    assert result[
        "risk_level"
    ] == "high"


def test_type_consistency_preserves_missing_information():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
        values=[
            "A",
            None,
            "B",
            pd.NA,
        ],
    ).result

    assert result[
        "total_count"
    ] == 4

    assert result[
        "non_missing_count"
    ] == 2

    assert result[
        "missing_count"
    ] == 2


def test_type_consistency_reports_all_missing():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
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
        "type_families"
    ] == []

    assert result[
        "risk_level"
    ] == "critical"


def test_type_consistency_does_not_treat_boolean_as_numeric():
    result = DataAnalysisEngine().analyze(
        "type_consistency_diagnostic",
        values=[
            True,
            False,
            True,
        ],
    ).result

    assert result[
        "type_families"
    ] == ["boolean"]

    assert result[
        "recommended_target_type"
    ] == "boolean"
