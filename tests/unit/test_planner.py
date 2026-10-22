import pandas as pd

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.datasets import DatasetLoader
from data_analysis_engine.orchestration import (
    AnalysisPlanner,
    AnalyticalRouter,
)
from data_analysis_engine.profiling import DataProfiler


def build_plan(dataframe: pd.DataFrame):
    DataAnalysisEngine()

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)
    routing_report = AnalyticalRouter().route(profile)

    return AnalysisPlanner().plan(routing_report)


def test_per_column_method_creates_one_task_per_measure():
    plan = build_plan(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 4, 7, 10],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    tasks = [
        task
        for task in plan.tasks
        if task.method == "descriptive_statistics"
    ]

    assert len(tasks) == 2

    selected_columns = {
        task.columns[0]
        for task in tasks
    }

    assert selected_columns == {
        "sales",
        "profit",
    }


def test_pairwise_method_creates_all_unique_pairs():
    plan = build_plan(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 5, 7, 11],
                "cost": [8, 15, 22, 27],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    tasks = [
        task
        for task in plan.tasks
        if task.method == "pearson_correlation"
    ]

    assert len(tasks) == 3

    pairs = {
        tuple(task.columns)
        for task in tasks
    }

    assert pairs == {
        ("sales", "profit"),
        ("sales", "cost"),
        ("profit", "cost"),
    }


def test_spearman_uses_same_pairwise_planning_rule():
    plan = build_plan(
        pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2, 5, 6, 9],
                "c": [7, 3, 8, 10],
            }
        )
    )

    tasks = [
        task
        for task in plan.tasks
        if task.method == "spearman_correlation"
    ]

    assert len(tasks) == 3


def test_joint_method_creates_one_multicolumn_task():
    plan = build_plan(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 4, 7, 10],
                "cost": [8, 16, 22, 30],
            }
        )
    )

    covariance_tasks = [
        task
        for task in plan.tasks
        if task.method == "covariance_matrix"
    ]

    assert len(covariance_tasks) == 1

    assert set(
        covariance_tasks[0].columns
    ) == {
        "sales",
        "profit",
        "cost",
    }


def test_identifier_never_appears_in_analysis_tasks():
    plan = build_plan(
        pd.DataFrame(
            {
                "customer_id": [101, 102, 103, 104],
                "sales": [10, 20, 30, 40],
                "profit": [1, 3, 6, 8],
            }
        )
    )

    for task in plan.tasks:
        assert "customer_id" not in task.columns


def test_plan_preserves_dataset_fingerprint():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30],
            "profit": [2, 5, 8],
        }
    )

    DataAnalysisEngine()

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)
    routing_report = AnalyticalRouter().route(profile)
    plan = AnalysisPlanner().plan(routing_report)

    assert plan.dataset_fingerprint == dataset.fingerprint()
