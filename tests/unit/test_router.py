import pandas as pd

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.datasets import DatasetLoader
from data_analysis_engine.orchestration import AnalyticalRouter
from data_analysis_engine.profiling import DataProfiler


def build_profile(dataframe: pd.DataFrame):
    dataset = DatasetLoader.from_dataframe(dataframe)
    return DataProfiler().profile(dataset)


def test_descriptive_statistics_is_recommended_for_measure():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "region": ["North", "South", "North", "South"],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    methods = {
        candidate.method
        for candidate in report.candidates
    }

    assert "descriptive_statistics" in methods


def test_identifier_is_not_used_as_measure():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    methods = {
        candidate.method
        for candidate in report.candidates
    }

    assert "descriptive_statistics" not in methods


def test_router_returns_compatible_columns():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 4, 8, 10],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidate = next(
        item
        for item in report.candidates
        if item.method == "descriptive_statistics"
    )

    assert set(candidate.compatible_columns) == {
        "sales",
        "profit",
    }

    assert "customer_id" not in candidate.compatible_columns


def test_router_explains_rejection():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "customer_id": [101, 102, 103],
                "comment": [
                    "First comment",
                    "Second comment",
                    "Third comment",
                ],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    rejected_names = {
        item.method
        for item in report.rejected
    }

    assert "descriptive_statistics" in rejected_names


def test_router_discovers_pairwise_correlations_automatically():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 5, 7, 11],
                "customer_id": [101, 102, 103, 104],
                "region": [
                    "North",
                    "South",
                    "North",
                    "South",
                ],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    assert "pearson_correlation" in candidates
    assert "spearman_correlation" in candidates

    assert (
        candidates["pearson_correlation"].selection_mode
        == "pairwise"
    )

    assert set(
        candidates["pearson_correlation"].compatible_columns
    ) == {
        "sales",
        "profit",
    }

    assert "customer_id" not in (
        candidates["pearson_correlation"].compatible_columns
    )


def test_correlations_require_at_least_two_measures():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "region": [
                    "North",
                    "South",
                    "North",
                    "South",
                ],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    rejected = {
        rejection.method
        for rejection in report.rejected
    }

    assert "pearson_correlation" in rejected
    assert "spearman_correlation" in rejected
