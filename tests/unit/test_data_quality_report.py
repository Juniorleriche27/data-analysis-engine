import numpy as np
import pandas as pd

from data_analysis_engine import DataAnalysisEngine


def test_data_quality_report_returns_dataset_summary():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "sales": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ],
            "cost": [
                8.0,
                12.0,
                21.0,
                19.0,
                37.0,
                31.0,
            ],
            "category": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
        },
    ).result

    assert result[
        "row_count"
    ] == 6

    assert result[
        "column_count"
    ] == 3

    assert set(
        result[
            "columns"
        ]
    ) == {
        "sales",
        "cost",
        "category",
    }

    assert result[
        "executed_check_count"
    ] > 0


def test_data_quality_report_contains_column_diagnostics():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "sales": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "category": [
                "A",
                "B",
                "A",
                "B",
            ],
        },
    ).result

    sales_checks = result[
        "column_reports"
    ]["sales"]["checks"]

    assert "missingness" in sales_checks
    assert "constantness" in sales_checks
    assert "type_consistency" in sales_checks
    assert "non_finite" in sales_checks
    assert "outliers" in sales_checks
    assert "variance" in sales_checks

    category_checks = result[
        "column_reports"
    ]["category"]["checks"]

    assert "cardinality" in category_checks


def test_data_quality_report_detects_missing_quality_issue():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "sales": [
                10.0,
                None,
                None,
                40.0,
            ],
        },
    ).result

    missing = result[
        "column_reports"
    ]["sales"]["checks"][
        "missingness"
    ]["result"]

    assert missing[
        "missing_count"
    ] == 2

    assert result[
        "issue_count"
    ] > 0


def test_data_quality_report_contains_duplicate_check():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "sales": [
                10,
                10,
                20,
                30,
            ],
            "category": [
                "A",
                "A",
                "B",
                "C",
            ],
        },
    ).result

    duplicate = result[
        "dataset_checks"
    ]["duplicate_rows"]

    assert duplicate[
        "status"
    ] == "success"

    assert duplicate[
        "result"
    ][
        "excess_duplicate_count"
    ] == 1


def test_data_quality_report_contains_temporal_check():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=5,
                freq="h",
            ),
        },
    ).result

    checks = result[
        "column_reports"
    ]["timestamp"]["checks"]

    assert (
        "temporal_quality"
        in checks
    )

    assert checks[
        "temporal_quality"
    ]["result"][
        "quality_status"
    ] == "clean"


def test_data_quality_report_contains_multicollinearity_check():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "x1": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "x2": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                12.0,
            ],
        },
    ).result

    check = result[
        "dataset_checks"
    ][
        "multicollinearity"
    ]

    assert check[
        "status"
    ] == "success"

    assert check[
        "result"
    ][
        "has_multicollinearity"
    ] is True


def test_data_quality_report_handles_non_finite_without_failing_report():
    result = DataAnalysisEngine().analyze(
        "data_quality_report",
        data={
            "sales": [
                1.0,
                2.0,
                np.inf,
                4.0,
            ],
        },
    ).result

    checks = result[
        "column_reports"
    ]["sales"]["checks"]

    assert checks[
        "non_finite"
    ]["status"] == "success"

    assert checks[
        "outliers"
    ]["status"] == "skipped"

    assert checks[
        "variance"
    ]["status"] == "skipped"
