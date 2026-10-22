from pathlib import Path

import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisPipeline


def test_pipeline_analyzes_dataframe_end_to_end():
    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "sales": [10, 20, 30, 40],
            "profit": [2, 4, 6, 8],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        name="sales_test",
    )

    assert report.dataset_name == "sales_test"
    assert report.profile.row_count == 4
    assert report.profile.column_count == 3
    assert len(report.plan.tasks) > 0
    assert report.execution.succeeded > 0


def test_pipeline_executes_correlations_automatically():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40],
            "profit": [2, 4, 6, 8],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe
    )

    pearson = next(
        item
        for item in report.execution.results
        if item.method == "pearson_correlation"
    )

    assert pearson.status == "success"

    assert (
        pearson.result["result"]["coefficient"]
        == pytest.approx(1.0)
    )


def test_pipeline_accepts_method_parameters():
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [2, 4, 6],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        parameters_by_method={
            "linear_combination_variance": {
                "weights": {
                    "x": 0.5,
                    "y": 0.5,
                }
            }
        },
    )

    weighted = next(
        item
        for item in report.execution.results
        if item.method == "linear_combination_variance"
    )

    assert weighted.status == "success"

    assert (
        weighted.result["result"]["variance"]
        == pytest.approx(2.25)
    )


def test_pipeline_report_serializes_to_dictionary():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30],
            "profit": [2, 5, 8],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe
    )

    serialized = report.to_dict()

    assert "dataset" in serialized
    assert "profile" in serialized
    assert "routing" in serialized
    assert "plan" in serialized
    assert "execution" in serialized


def test_pipeline_can_analyze_csv(tmp_path: Path):
    path = tmp_path / "data.csv"

    pd.DataFrame(
        {
            "sales": [10, 20, 30],
            "profit": [2, 4, 6],
        }
    ).to_csv(
        path,
        index=False,
    )

    report = DataAnalysisPipeline().analyze_csv(
        path
    )

    assert report.profile.row_count == 3
    assert report.execution.succeeded > 0


def test_identifier_is_excluded_end_to_end():
    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "sales": [10, 20, 30, 40],
            "profit": [2, 4, 6, 8],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe
    )

    for task in report.plan.tasks:
        assert "customer_id" not in task.columns
