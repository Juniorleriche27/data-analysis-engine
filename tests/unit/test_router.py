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


def test_router_discovers_volatility_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=5,
                    freq="D",
                ),
                "returns": [
                    0.01,
                    -0.02,
                    0.03,
                    -0.01,
                    0.02,
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

    assert "rolling_volatility" in candidates
    assert "ewma_volatility" in candidates


def test_router_discovers_mean_shift_change_point():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    1,
                    1,
                    1,
                    10,
                    10,
                    10,
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

    assert (
        "mean_shift_change_point"
        in candidates
    )


def test_router_discovers_variance_shift_change_point():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    -1,
                    1,
                    -1,
                    -10,
                    10,
                    -10,
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

    assert (
        "variance_shift_change_point"
        in candidates
    )


def test_router_discovers_temporal_anomaly_methods():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    10,
                    11,
                    10,
                    12,
                    50,
                    11,
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

    assert (
        "rolling_zscore_anomaly"
        in candidates
    )

    assert (
        "rolling_mad_anomaly"
        in candidates
    )


def test_router_discovers_temporal_structure_diagnostic():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=4,
                    freq="D",
                ),
                "value": [
                    10,
                    20,
                    30,
                    40,
                ],
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

    candidate = candidates[
        "temporal_structure_diagnostic"
    ]

    assert (
        candidate.selection_mode
        == "time_axis"
    )

    assert candidate.compatible_columns == [
        "date"
    ]


def test_router_discovers_forecasting_baselines():
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
                    14,
                    16,
                    18,
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

    assert "naive_forecast" in candidates
    assert "drift_forecast" in candidates


def test_router_discovers_forecast_backtest():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
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

    assert (
        "rolling_origin_forecast_backtest"
        in candidates
    )


def test_router_discovers_autoregressive_forecast():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=8,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    11,
                    14,
                    13,
                    16,
                    15,
                    18,
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

    assert (
        "autoregressive_forecast"
        in candidates
    )


def test_router_discovers_autoregressive_forecast():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=8,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    11,
                    14,
                    13,
                    16,
                    15,
                    18,
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

    assert (
        "autoregressive_forecast"
        in candidates
    )


def test_router_discovers_autoregressive_order_selection():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=10,
                    freq="D",
                ),
                "value": [
                    10,
                    11,
                    12,
                    14,
                    15,
                    17,
                    18,
                    20,
                    21,
                    23,
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

    assert (
        "autoregressive_order_selection"
        in candidates
    )


def test_router_discovers_seasonal_naive_forecast():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=8,
                    freq="D",
                ),
                "value": [
                    10,
                    20,
                    30,
                    10,
                    20,
                    30,
                    10,
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

    assert (
        "seasonal_naive_forecast"
        in candidates
    )


def test_router_discovers_simple_exponential_smoothing():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    11,
                    14,
                    13,
                    16,
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

    assert (
        "simple_exponential_smoothing_forecast"
        in candidates
    )


def test_router_discovers_simple_exponential_smoothing():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    11,
                    14,
                    13,
                    16,
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

    assert (
        "simple_exponential_smoothing_forecast"
        in candidates
    )


def test_router_discovers_ses_alpha_selection():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=10,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    11,
                    14,
                    13,
                    16,
                    15,
                    18,
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

    assert (
        "simple_exponential_smoothing_alpha_selection"
        in candidates
    )


def test_router_discovers_holt_linear_trend_forecast():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
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

    assert (
        "holt_linear_trend_forecast"
        in candidates
    )


def test_router_discovers_holt_parameter_selection():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=10,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    17,
                    19,
                    22,
                    25,
                    27,
                    30,
                    33,
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

    assert (
        "holt_parameter_selection"
        in candidates
    )


def test_router_discovers_holt_winters_additive_forecast():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    110,
                    97,
                    109,
                    96,
                    118,
                    105,
                    117,
                    104,
                    126,
                    113,
                    125,
                    112,
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

    assert (
        "holt_winters_additive_forecast"
        in candidates
    )


def test_router_discovers_holt_winters_parameter_selection():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    110,
                    97,
                    109,
                    96,
                    118,
                    105,
                    117,
                    104,
                    126,
                    113,
                    125,
                    112,
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

    assert (
        "holt_winters_additive_parameter_selection"
        in candidates
    )


def test_router_discovers_empirical_forecast_interval():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=10,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
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

    assert (
        "empirical_forecast_interval"
        in candidates
    )


def test_router_discovers_empirical_forecast_interval():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=10,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
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

    assert (
        "empirical_forecast_interval"
        in candidates
    )


def test_router_discovers_empirical_forecast_interval():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=10,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
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

    assert (
        "empirical_forecast_interval"
        in candidates
    )


def test_router_discovers_forecast_interval_calibration():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
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

    assert (
        "forecast_interval_calibration"
        in candidates
    )


def test_router_discovers_conformal_forecast_interval():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
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

    assert (
        "conformal_forecast_interval"
        in candidates
    )


def test_router_discovers_conformal_interval_calibration():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
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

    assert (
        "conformal_interval_calibration"
        in candidates
    )


def test_router_discovers_forecast_model_selection():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
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

    assert (
        "forecast_model_selection"
        in candidates
    )


def test_router_discovers_automatic_forecast():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                ],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method
        for item
        in report.candidates
    }

    assert (
        "automatic_forecast"
        in candidates
    )


def test_router_discovers_automatic_forecast_with_intervals():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                ],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method
        for item
        in report.candidates
    }

    assert (
        "automatic_forecast_with_intervals"
        in candidates
    )


def test_router_discovers_automatic_forecast_reliability():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=14,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                    34,
                    36,
                ],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method
        for item
        in report.candidates
    }

    assert (
        "automatic_forecast_reliability"
        in candidates
    )


def test_router_discovers_weighted_forecast_ensemble():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=12,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                ],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method
        for item
        in report.candidates
    }

    assert (
        "weighted_forecast_ensemble"
        in candidates
    )


def test_router_discovers_adaptive_forecast_ensemble_backtest():
    DataAnalysisEngine()

    profile = build_profile(
        pd.DataFrame(
            {
                "date": pd.date_range(
                    "2026-01-01",
                    periods=14,
                    freq="D",
                ),
                "value": [
                    10,
                    12,
                    14,
                    16,
                    18,
                    20,
                    22,
                    24,
                    26,
                    28,
                    30,
                    32,
                    34,
                    36,
                ],
            }
        )
    )

    report = AnalyticalRouter().route(
        profile
    )

    candidates = {
        item.method
        for item
        in report.candidates
    }

    assert (
        "adaptive_forecast_ensemble_backtest"
        in candidates
    )
