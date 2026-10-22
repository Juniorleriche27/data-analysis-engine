import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    PlanDatasetMismatchError,
)
from data_analysis_engine.datasets import DatasetLoader
from data_analysis_engine.orchestration import (
    AnalysisExecutor,
    AnalysisPlanner,
    AnalyticalRouter,
)
from data_analysis_engine.profiling import DataProfiler


def build_dataset_and_plan(
    dataframe: pd.DataFrame,
):
    DataAnalysisEngine()

    dataset = DatasetLoader.from_dataframe(
        dataframe
    )

    profile = DataProfiler().profile(dataset)

    routing = AnalyticalRouter().route(
        profile
    )

    plan = AnalysisPlanner().plan(
        routing
    )

    return dataset, plan


def test_executor_runs_descriptive_statistics():
    dataset, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [1, 2, 4, 8],
            }
        )
    )

    report = AnalysisExecutor().execute(
        dataset,
        plan,
    )

    sales_result = next(
        item
        for item in report.results
        if (
            item.method == "descriptive_statistics"
            and item.columns == ["sales"]
        )
    )

    assert sales_result.status == "success"

    assert (
        sales_result.result["result"]["mean"]
        == pytest.approx(25.0)
    )


def test_executor_runs_pairwise_correlation():
    dataset, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 4, 6, 8],
            }
        )
    )

    report = AnalysisExecutor().execute(
        dataset,
        plan,
    )

    pearson = next(
        item
        for item in report.results
        if item.method == "pearson_correlation"
    )

    assert pearson.status == "success"

    assert (
        pearson.result["result"]["coefficient"]
        == pytest.approx(1.0)
    )


def test_executor_drops_missing_values_pairwise():
    dataset, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "sales": [
                    10,
                    20,
                    None,
                    40,
                    50,
                ],
                "profit": [
                    2,
                    None,
                    6,
                    8,
                    10,
                ],
            }
        )
    )

    report = AnalysisExecutor().execute(
        dataset,
        plan,
    )

    pearson = next(
        item
        for item in report.results
        if item.method == "pearson_correlation"
    )

    assert pearson.status == "success"
    assert pearson.rows_used == 3
    assert pearson.rows_dropped == 2

    assert (
        pearson.result["result"]["sample_size"]
        == 3
    )


def test_executor_runs_covariance_matrix():
    dataset, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )
    )

    report = AnalysisExecutor().execute(
        dataset,
        plan,
    )

    covariance = next(
        item
        for item in report.results
        if item.method == "covariance_matrix"
    )

    assert covariance.status == "success"

    matrix = (
        covariance.result["result"]
        ["covariance_matrix"]
    )

    assert matrix[0][1] == pytest.approx(2.0)


def test_executor_accepts_weights_for_linear_combination():
    dataset, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )
    )

    report = AnalysisExecutor().execute(
        dataset,
        plan,
        parameters_by_method={
            "linear_combination_variance": {
                "weights": {
                    "x": 0.5,
                    "y": 0.5,
                }
            }
        },
    )

    result = next(
        item
        for item in report.results
        if (
            item.method
            == "linear_combination_variance"
        )
    )

    assert result.status == "success"

    assert (
        result.result["result"]["variance"]
        == pytest.approx(2.25)
    )


def test_missing_weights_fail_only_that_task():
    dataset, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )
    )

    report = AnalysisExecutor().execute(
        dataset,
        plan,
    )

    weighted = next(
        item
        for item in report.results
        if (
            item.method
            == "linear_combination_variance"
        )
    )

    covariance = next(
        item
        for item in report.results
        if item.method == "covariance_matrix"
    )

    assert weighted.status == "failed"
    assert covariance.status == "success"


def test_plan_cannot_run_on_different_dataset():
    dataset_a, plan = build_dataset_and_plan(
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 4, 6],
            }
        )
    )

    dataset_b = DatasetLoader.from_dataframe(
        pd.DataFrame(
            {
                "x": [10, 20, 30],
                "y": [1, 5, 9],
            }
        )
    )

    with pytest.raises(
        PlanDatasetMismatchError
    ):
        AnalysisExecutor().execute(
            dataset_b,
            plan,
        )
