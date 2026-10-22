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


def test_router_discovers_multivariate_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [3, 4, 8, 15],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    assert "covariance_matrix" in candidates
    assert "linear_combination_variance" in candidates

    assert (
        candidates["covariance_matrix"].selection_mode
        == "joint"
    )

    assert set(
        candidates["covariance_matrix"].compatible_columns
    ) == {
        "sales",
        "profit",
    }


def test_multivariate_methods_require_multiple_measures():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    rejected = {
        rejection.method
        for rejection in report.rejected
    }

    assert "covariance_matrix" in rejected
    assert "linear_combination_variance" in rejected


def test_router_discovers_numeric_similarity_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [3, 6, 8, 15],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    expected_methods = {
        "euclidean_distance",
        "manhattan_distance",
        "minkowski_distance",
        "cosine_similarity",
        "cosine_distance",
    }

    assert expected_methods.issubset(
        candidates.keys()
    )

    for method in expected_methods:
        assert set(
            candidates[method].compatible_columns
        ) == {
            "sales",
            "profit",
        }

        assert (
            candidates[method].selection_mode
            == "pairwise"
        )

        assert (
            "customer_id"
            not in candidates[
                method
            ].compatible_columns
        )


def test_router_discovers_numeric_similarity_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [3, 6, 8, 15],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    expected_methods = {
        "euclidean_distance",
        "manhattan_distance",
        "minkowski_distance",
        "cosine_similarity",
        "cosine_distance",
    }

    assert expected_methods.issubset(
        candidates.keys()
    )

    for method in expected_methods:
        assert set(
            candidates[method].compatible_columns
        ) == {
            "sales",
            "profit",
        }

        assert (
            candidates[method].selection_mode
            == "pairwise"
        )

        assert (
            "customer_id"
            not in candidates[
                method
            ].compatible_columns
        )


def test_router_discovers_numeric_similarity_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [3, 6, 8, 15],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    expected_methods = {
        "euclidean_distance",
        "manhattan_distance",
        "minkowski_distance",
        "cosine_similarity",
        "cosine_distance",
    }

    assert expected_methods.issubset(
        candidates.keys()
    )

    for method in expected_methods:
        assert set(
            candidates[method].compatible_columns
        ) == {
            "sales",
            "profit",
        }

        assert (
            candidates[method].selection_mode
            == "pairwise"
        )

        assert (
            "customer_id"
            not in candidates[method].compatible_columns
        )


def test_router_discovers_pca_automatically():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [3, 7, 8, 14],
                "cost": [8, 13, 23, 29],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    assert (
        "principal_component_analysis"
        in candidates
    )

    pca = candidates[
        "principal_component_analysis"
    ]

    assert pca.selection_mode == "joint"

    assert set(
        pca.compatible_columns
    ) == {
        "sales",
        "profit",
        "cost",
    }

    assert (
        "customer_id"
        not in pca.compatible_columns
    )


def test_router_discovers_pca_automatically():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [3, 7, 8, 14],
                "cost": [8, 13, 23, 29],
                "customer_id": [101, 102, 103, 104],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    assert (
        "principal_component_analysis"
        in candidates
    )

    pca = candidates[
        "principal_component_analysis"
    ]

    assert pca.selection_mode == "joint"

    assert set(
        pca.compatible_columns
    ) == {
        "sales",
        "profit",
        "cost",
    }

    assert (
        "customer_id"
        not in pca.compatible_columns
    )


def test_router_discovers_time_series_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                    ]
                ),
                "sales": [10, 20, 30, 40],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    expected = {
        "simple_returns",
        "log_returns",
        "rolling_statistics",
        "autocorrelation",
    }

    assert expected.issubset(
        candidates.keys()
    )

    candidate = candidates[
        "simple_returns"
    ]

    assert candidate.role_requirements == {
        "time": 1,
        "measure": 1,
    }


def test_router_rejects_time_series_without_time_column():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "sales": [10, 20, 30, 40],
                "profit": [2, 4, 6, 8],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    rejected = {
        item.method
        for item in report.rejected
    }

    assert "simple_returns" in rejected
    assert "autocorrelation" in rejected


def test_router_discovers_extended_time_series_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                    ]
                ),
                "sales": [10, 20, 30, 40],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        candidate.method
        for candidate in report.candidates
    }

    assert "lagged_series" in candidates
    assert "autocorrelation_function" in candidates


def test_router_discovers_dtw_with_time_and_two_measures():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                    ]
                ),
                "sales": [10, 20, 30, 40],
                "profit": [2, 5, 8, 11],
            }
        )
    )

    report = AnalyticalRouter().route(profile)

    candidates = {
        candidate.method: candidate
        for candidate in report.candidates
    }

    assert "dynamic_time_warping" in candidates

    dtw = candidates[
        "dynamic_time_warping"
    ]

    assert dtw.role_requirements == {
        "time": 1,
        "measure": 2,
    }


def test_router_discovers_cross_correlation():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                    ]
                ),
                "sales": [10, 20, 30, 40],
                "profit": [3, 5, 9, 12],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method: item
        for item in report.candidates
    }

    assert "cross_correlation" in candidates

    assert candidates[
        "cross_correlation"
    ].selection_mode == "time_series_pairwise"


def test_router_discovers_linear_trend_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=5,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    15,
                    17,
                    20,
                ],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method
        for item in report.candidates
    }

    assert "linear_trend" in candidates
    assert "linear_detrend" in candidates
