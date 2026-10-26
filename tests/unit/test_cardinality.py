import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_cardinality_diagnostic_reports_normal_cardinality():
    result = DataAnalysisEngine().analyze(
        "cardinality_diagnostic",
        values=[
            "A",
            "A",
            "B",
            "B",
            "C",
            "C",
        ],
    ).result

    assert result["unique_count"] == 3
    assert result["unique_ratio"] == pytest.approx(0.5)
    assert result["singleton_count"] == 0
    assert result["cardinality_status"] == "normal"
    assert result["risk_level"] == "none"


def test_cardinality_diagnostic_detects_unique_column():
    result = DataAnalysisEngine().analyze(
        "cardinality_diagnostic",
        values=[
            "A",
            "B",
            "C",
            "D",
        ],
    ).result

    assert result["unique_count"] == 4
    assert result["unique_ratio"] == pytest.approx(1.0)
    assert result["is_unique"] is True
    assert result["cardinality_status"] == "unique"
    assert result["risk_level"] == "critical"


def test_cardinality_diagnostic_detects_high_cardinality():
    values = [
        f"group_{index}"
        for index in range(90)
    ] + ["shared"] * 10

    result = DataAnalysisEngine().analyze(
        "cardinality_diagnostic",
        values=values,
        high_cardinality_ratio=0.90,
        minimum_unique_count=20,
    ).result

    assert result["unique_count"] == 91
    assert result["unique_ratio"] == pytest.approx(0.91)
    assert result["is_unique"] is False
    assert result["is_high_cardinality"] is True
    assert result["cardinality_status"] == "high"
    assert result["risk_level"] == "high"


def test_cardinality_diagnostic_reports_singletons():
    result = DataAnalysisEngine().analyze(
        "cardinality_diagnostic",
        values=[
            "A",
            "A",
            "B",
            "C",
        ],
    ).result

    assert result["singleton_count"] == 2
    assert result["singleton_ratio"] == pytest.approx(0.5)
    assert result["dominant_value"] == "A"
    assert result["dominant_count"] == 2


def test_cardinality_diagnostic_preserves_missing_information():
    result = DataAnalysisEngine().analyze(
        "cardinality_diagnostic",
        values=[
            "A",
            None,
            "A",
            "B",
            pd.NA,
        ],
    ).result

    assert result["total_count"] == 5
    assert result["non_missing_count"] == 3
    assert result["missing_count"] == 2
    assert result["unique_count"] == 2


def test_cardinality_diagnostic_reports_all_missing():
    result = DataAnalysisEngine().analyze(
        "cardinality_diagnostic",
        values=[
            None,
            pd.NA,
            None,
        ],
    ).result

    assert result["is_all_missing"] is True
    assert result["unique_count"] == 0
    assert result["unique_ratio"] is None
    assert result["risk_level"] == "critical"


def test_cardinality_diagnostic_rejects_invalid_thresholds():
    engine = DataAnalysisEngine()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        engine.analyze(
            "cardinality_diagnostic",
            values=["A", "B"],
            high_cardinality_ratio=0.0,
        )

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        engine.analyze(
            "cardinality_diagnostic",
            values=["A", "B"],
            minimum_unique_count=1,
        )
