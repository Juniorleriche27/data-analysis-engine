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


def test_pipeline_executes_pca_automatically():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40],
            "profit": [2, 5, 8, 12],
            "cost": [8, 15, 22, 28],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe
    )

    pca = next(
        item
        for item in report.execution.results
        if (
            item.method
            == "principal_component_analysis"
        )
    )

    assert pca.status == "success"

    assert (
        pca.result["result"][
            "variable_count"
        ]
        == 3
    )

    assert (
        sum(
            pca.result["result"][
                "explained_variance_ratio"
            ]
        )
        == pytest.approx(1.0)
    )


def test_pipeline_can_execute_only_selected_methods():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40],
            "profit": [2, 5, 8, 11],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "pearson_correlation",
        ],
    )

    assert len(
        report.execution.results
    ) == 1

    assert (
        report.execution.results[0].method
        == "pearson_correlation"
    )


def test_pipeline_can_select_multiple_methods():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40],
            "profit": [2, 5, 8, 11],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "pearson_correlation",
            "spearman_correlation",
        ],
    )

    executed = {
        item.method
        for item in report.execution.results
    }

    assert executed == {
        "pearson_correlation",
        "spearman_correlation",
    }


def test_pipeline_rejects_unknown_method():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30],
            "profit": [2, 5, 8],
        }
    )

    with pytest.raises(Exception):
        DataAnalysisPipeline().analyze_dataframe(
            dataframe,
            methods=[
                "method_that_does_not_exist",
            ],
        )


def test_pipeline_preserves_default_all_methods_behavior():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40],
            "profit": [2, 5, 8, 11],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe
    )

    executed = {
        item.method
        for item in report.execution.results
    }

    assert "pearson_correlation" in executed
    assert "spearman_correlation" in executed
    assert "principal_component_analysis" in executed


def test_pipeline_executes_pca_automatically():
    dataframe = pd.DataFrame(
        {
            "sales": [10, 20, 30, 40],
            "profit": [2, 5, 8, 12],
            "cost": [8, 15, 22, 28],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe
    )

    pca = next(
        item
        for item in report.execution.results
        if (
            item.method
            == "principal_component_analysis"
        )
    )

    assert pca.status == "success"

    assert (
        pca.result["result"][
            "variable_count"
        ]
        == 3
    )

    assert (
        sum(
            pca.result["result"][
                "explained_variance_ratio"
            ]
        )
        == pytest.approx(1.0)
    )


def test_pipeline_executes_time_series_in_chronological_order():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "price": [
                121,
                100,
                110,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "simple_returns",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    returns = result.result[
        "result"
    ]["returns"]

    assert returns == pytest.approx(
        [
            0.10,
            0.10,
        ]
    )


def test_pipeline_executes_extended_time_series_methods():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ]
            ),
            "price": [
                100,
                103,
                107,
                106,
                112,
                118,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "lagged_series",
            "autocorrelation_function",
        ],
        parameters_by_method={
            "lagged_series": {
                "lag": 2,
            },
            "autocorrelation_function": {
                "max_lag": 3,
            },
        },
    )

    executed = {
        item.method: item
        for item in report.execution.results
    }

    assert executed[
        "lagged_series"
    ].status == "success"

    assert executed[
        "autocorrelation_function"
    ].status == "success"


def test_pipeline_executes_cross_correlation():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=8,
                freq="D",
            ),
            "x": [
                1,
                4,
                2,
                8,
                3,
                7,
                5,
                9,
            ],
            "y": [
                0,
                1,
                4,
                2,
                8,
                3,
                7,
                5,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "cross_correlation",
        ],
        parameters_by_method={
            "cross_correlation": {
                "max_lag": 2,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    strongest = result.result[
        "result"
    ]["strongest_relationship"]

    assert strongest["lag"] == 1

    assert (
        strongest["direction"]
        == "x_leads_y"
    )


def test_pipeline_executes_differencing_methods():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="D",
            ),
            "value": [
                10,
                20,
                30,
                15,
                25,
                35,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "first_difference",
            "seasonal_difference",
        ],
        parameters_by_method={
            "seasonal_difference": {
                "period": 3,
            }
        },
    )

    results = {
        item.method: item
        for item in report.execution.results
    }

    assert results[
        "first_difference"
    ].status == "success"

    assert results[
        "seasonal_difference"
    ].status == "success"

    seasonal = results[
        "seasonal_difference"
    ].result["result"]

    assert seasonal[
        "differences"
    ] == pytest.approx(
        [
            5,
            5,
            5,
        ]
    )


def test_pipeline_executes_additive_decomposition():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=12,
                freq="D",
            ),
            "value": [
                12,
                10,
                10,
                14,
                16,
                14,
                14,
                18,
                20,
                18,
                18,
                22,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "additive_seasonal_decomposition",
        ],
        parameters_by_method={
            "additive_seasonal_decomposition": {
                "period": 4,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    decomposition = result.result[
        "result"
    ]

    assert decomposition[
        "model"
    ] == "additive"

    assert decomposition[
        "period"
    ] == 4


def test_pipeline_executes_linear_trend_and_detrend():
    dataframe = pd.DataFrame(
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

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "linear_trend",
            "linear_detrend",
        ],
    )

    results = {
        item.method: item
        for item in report.execution.results
    }

    assert results[
        "linear_trend"
    ].status == "success"

    assert results[
        "linear_detrend"
    ].status == "success"

    assert results[
        "linear_trend"
    ].result["result"][
        "slope"
    ] == pytest.approx(2.0)


def test_pipeline_executes_stability_diagnostic():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=10,
                freq="D",
            ),
            "value": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "time_series_stability_diagnostic",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert (
        diagnostic[
            "formal_stationarity_test"
        ]
        is False
    )

    assert (
        diagnostic["assessment"]
        == "instability_signals_detected"
    )


def test_pipeline_executes_mean_shift_change_point():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-06",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-07",
                    "2026-01-08",
                ]
            ),
            "value": [
                10,
                0,
                0,
                0,
                0,
                10,
                10,
                10,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "mean_shift_change_point",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "detected"
    ] is True

    assert diagnostic[
        "change_timestamp"
    ] == "2026-01-05T00:00:00"


def test_pipeline_executes_variance_shift_change_point():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=12,
                freq="D",
            ),
            "value": [
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -10,
                10,
                -10,
                10,
                -10,
                10,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "variance_shift_change_point",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "detected"
    ] is True

    assert diagnostic[
        "change_timestamp"
    ] == "2026-01-07T00:00:00"

    assert diagnostic[
        "direction"
    ] == "variance_increase"


def test_pipeline_executes_temporal_anomaly_detection():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="D",
            ),
            "value": [
                10,
                10,
                10,
                10,
                50,
                10,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "rolling_zscore_anomaly",
            "rolling_mad_anomaly",
        ],
        parameters_by_method={
            "rolling_zscore_anomaly": {
                "window": 3,
            },
            "rolling_mad_anomaly": {
                "window": 3,
            },
        },
    )

    results = {
        item.method: item
        for item in report.execution.results
    }

    assert results[
        "rolling_zscore_anomaly"
    ].status == "success"

    assert results[
        "rolling_mad_anomaly"
    ].status == "success"

    assert results[
        "rolling_zscore_anomaly"
    ].result["result"][
        "anomaly_count"
    ] == 1

    assert results[
        "rolling_mad_anomaly"
    ].result["result"][
        "anomaly_count"
    ] == 1


def test_pipeline_executes_temporal_structure_diagnostic():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-04",
                ]
            ),
            "value": [
                30,
                10,
                20,
                40,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "temporal_structure_diagnostic",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    assert result.columns == [
        "date"
    ]

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "input_monotonic_increasing"
    ] is False

    assert diagnostic[
        "status"
    ] == "regular"


def test_pipeline_executes_forecasting_baselines():
    dataframe = pd.DataFrame(
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

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "naive_forecast",
            "drift_forecast",
        ],
        parameters_by_method={
            "naive_forecast": {
                "horizon": 2,
            },
            "drift_forecast": {
                "horizon": 2,
            },
        },
    )

    results = {
        item.method: item
        for item in report.execution.results
    }

    assert results[
        "naive_forecast"
    ].status == "success"

    assert results[
        "drift_forecast"
    ].status == "success"

    assert results[
        "naive_forecast"
    ].result["result"][
        "forecast_values"
    ] == pytest.approx(
        [
            18,
            18,
        ]
    )

    assert results[
        "drift_forecast"
    ].result["result"][
        "forecast_values"
    ] == pytest.approx(
        [
            20,
            22,
        ]
    )


def test_pipeline_executes_forecast_backtest():
    dataframe = pd.DataFrame(
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

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "rolling_origin_forecast_backtest",
        ],
        parameters_by_method={
            "rolling_origin_forecast_backtest": {
                "model": "drift",
                "initial_train_size": 3,
                "horizon": 1,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    backtest = result.result[
        "result"
    ]

    assert (
        backtest[
            "look_ahead_leakage"
        ]
        is False
    )

    assert backtest[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0
    )


def test_pipeline_executes_autoregressive_forecast():
    values = [
        10.0
    ]

    for _ in range(7):
        values.append(
            2.0
            + 0.5
            * values[-1]
        )

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "autoregressive_forecast",
        ],
        parameters_by_method={
            "autoregressive_forecast": {
                "p": 1,
                "horizon": 2,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "coefficients"
    ]["lag_1"] == pytest.approx(
        0.5
    )

    assert len(
        forecast[
            "forecast_values"
        ]
    ) == 2


def test_pipeline_executes_autoregressive_backtest():
    values = [
        10.0
    ]

    for _ in range(11):
        values.append(
            2.0
            + 0.5
            * values[-1]
        )

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "rolling_origin_forecast_backtest",
        ],
        parameters_by_method={
            "rolling_origin_forecast_backtest": {
                "model": "autoregressive",
                "p": 1,
                "initial_train_size": 5,
                "horizon": 1,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    backtest = result.result[
        "result"
    ]

    assert (
        backtest["model"]
        == "autoregressive"
    )

    assert backtest[
        "model_parameters"
    ]["p"] == 1

    assert backtest[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_pipeline_executes_autoregressive_order_selection():
    values = [
        100,
        103,
        106,
        110,
        113,
        117,
        120,
        124,
        128,
        131,
        135,
        139,
        142,
        146,
        150,
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "autoregressive_order_selection",
        ],
        parameters_by_method={
            "autoregressive_order_selection": {
                "min_p": 1,
                "max_p": 3,
                "initial_train_size": 8,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    selection = result.result[
        "result"
    ]

    assert selection[
        "best_order"
    ] in {
        1,
        2,
        3,
    }

    assert selection[
        "orders_evaluated"
    ] == 3

    assert (
        selection[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_seasonal_naive_forecast():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=6,
                freq="D",
            ),
            "value": [
                10,
                20,
                30,
                10,
                20,
                30,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "seasonal_naive_forecast",
        ],
        parameters_by_method={
            "seasonal_naive_forecast": {
                "season_length": 3,
                "horizon": 3,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "model"
    ] == "seasonal_naive"

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            10,
            20,
            30,
        ]
    )


def test_pipeline_executes_simple_exponential_smoothing():
    dataframe = pd.DataFrame(
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

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "simple_exponential_smoothing_forecast",
        ],
        parameters_by_method={
            "simple_exponential_smoothing_forecast": {
                "alpha": 0.5,
                "horizon": 2,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "final_level"
    ] == pytest.approx(
        31.25
    )

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            31.25,
            31.25,
        ]
    )


def test_pipeline_executes_simple_exponential_smoothing():
    dataframe = pd.DataFrame(
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

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "simple_exponential_smoothing_forecast",
        ],
        parameters_by_method={
            "simple_exponential_smoothing_forecast": {
                "alpha": 0.5,
                "horizon": 2,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "final_level"
    ] == pytest.approx(
        31.25
    )

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            31.25,
            31.25,
        ]
    )


def test_pipeline_executes_ses_alpha_selection():
    values = [
        100,
        104,
        102,
        106,
        105,
        109,
        108,
        112,
        111,
        115,
        114,
        118,
        117,
        121,
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "simple_exponential_smoothing_alpha_selection",
        ],
        parameters_by_method={
            "simple_exponential_smoothing_alpha_selection": {
                "alphas": [
                    0.2,
                    0.5,
                    0.8,
                ],
                "initial_train_size": 7,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    selection = result.result[
        "result"
    ]

    assert selection[
        "best_alpha"
    ] == pytest.approx(
        0.8
    )

    assert selection[
        "alphas_evaluated"
    ] == 3

    assert (
        selection[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_holt_linear_trend_forecast():
    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ),
            "value": [
                10,
                12,
                14,
                16,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "holt_linear_trend_forecast",
        ],
        parameters_by_method={
            "holt_linear_trend_forecast": {
                "alpha": 1.0,
                "beta": 1.0,
                "horizon": 2,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            18.0,
            20.0,
        ]
    )


def test_pipeline_executes_holt_parameter_selection():
    values = [
        100,
        103,
        105,
        108,
        112,
        114,
        118,
        121,
        125,
        128,
        132,
        135,
        139,
        142,
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "holt_parameter_selection",
        ],
        parameters_by_method={
            "holt_parameter_selection": {
                "alphas": [
                    0.2,
                    0.5,
                    0.8,
                ],
                "betas": [
                    0.2,
                    0.5,
                    1.0,
                ],
                "initial_train_size": 7,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    selection = result.result[
        "result"
    ]

    assert selection[
        "best_alpha"
    ] == pytest.approx(
        0.5
    )

    assert selection[
        "best_beta"
    ] == pytest.approx(
        1.0
    )

    assert selection[
        "parameter_combinations_evaluated"
    ] == 9

    assert (
        selection[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_holt_winters_additive_forecast():
    values = [
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
        134,
        121,
        133,
        120,
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "holt_winters_additive_forecast",
        ],
        parameters_by_method={
            "holt_winters_additive_forecast": {
                "season_length": 4,
                "alpha": 1.0,
                "beta": 1.0,
                "gamma": 1.0,
                "horizon": 4,
            }
        },
    )

    result = report.execution.results[0]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            142.0,
            129.0,
            141.0,
            128.0,
        ]
    )


def test_pipeline_executes_holt_winters_parameter_selection():
    values = [
        110,
        97,
        109,
        96,
        118,
        105,
        117,
        104,
        127,
        112,
        126,
        111,
        133,
        122,
        134,
        119,
        143,
        128,
        142,
        129,
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "holt_winters_additive_parameter_selection",
        ],
        parameters_by_method={
            "holt_winters_additive_parameter_selection": {
                "season_length": 4,
                "alphas": [
                    0.5,
                    1.0,
                ],
                "betas": [
                    0.5,
                    1.0,
                ],
                "gammas": [
                    0.5,
                    1.0,
                ],
                "initial_train_size": 8,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    selection = result.result[
        "result"
    ]

    assert selection[
        "parameter_combinations_evaluated"
    ] == 8

    assert selection[
        "best_alpha"
    ] in {
        0.5,
        1.0,
    }

    assert selection[
        "best_beta"
    ] in {
        0.5,
        1.0,
    }

    assert selection[
        "best_gamma"
    ] in {
        0.5,
        1.0,
    }

    assert (
        selection[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_empirical_forecast_interval():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "empirical_forecast_interval",
        ],
        parameters_by_method={
            "empirical_forecast_interval": {
                "model": "drift",
                "initial_train_size": 5,
                "horizon": 2,
                "coverage": 0.8,
                "minimum_interval_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    interval = result.result[
        "result"
    ]

    assert interval[
        "forecast_values"
    ] == pytest.approx(
        [
            30.0,
            32.0,
        ]
    )

    assert len(
        interval[
            "intervals"
        ]
    ) == 2

    assert (
        interval[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_forecast_interval_calibration():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "forecast_interval_calibration",
        ],
        parameters_by_method={
            "forecast_interval_calibration": {
                "model": "drift",
                "initial_train_size": 5,
                "horizon": 1,
                "coverage": 0.9,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    calibration = result.result[
        "result"
    ]

    assert calibration[
        "empirical_coverage"
    ] == pytest.approx(
        1.0
    )

    assert calibration[
        "nominal_coverage"
    ] == pytest.approx(
        0.9
    )

    assert (
        calibration[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_conformal_forecast_interval():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "conformal_forecast_interval",
        ],
        parameters_by_method={
            "conformal_forecast_interval": {
                "model": "drift",
                "initial_train_size": 5,
                "horizon": 1,
                "coverage": 0.8,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    interval = result.result[
        "result"
    ]

    assert interval[
        "interval_type"
    ] == (
        "symmetric_finite_sample_conformal"
    )

    assert (
        interval[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_conformal_interval_calibration():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "conformal_interval_calibration",
        ],
        parameters_by_method={
            "conformal_interval_calibration": {
                "model": "drift",
                "initial_train_size": 5,
                "horizon": 1,
                "coverage": 0.9,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    calibration = result.result[
        "result"
    ]

    assert calibration[
        "empirical_coverage"
    ] == pytest.approx(
        1.0
    )

    assert calibration[
        "interval_type"
    ] == (
        "symmetric_finite_sample_conformal"
    )

    assert (
        calibration[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_forecast_model_selection():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "forecast_model_selection",
        ],
        parameters_by_method={
            "forecast_model_selection": {
                "models": [
                    "naive",
                    "drift",
                    "simple_exponential_smoothing",
                ],
                "parameters_by_model": {
                    "simple_exponential_smoothing": {
                        "alpha": 0.5,
                    }
                },
                "initial_train_size": 5,
                "horizon": 1,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    selection = result.result[
        "result"
    ]

    assert selection[
        "best_model"
    ] == "drift"

    assert selection[
        "best_score"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert (
        selection[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_automatic_forecast():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "automatic_forecast",
        ],
        parameters_by_method={
            "automatic_forecast": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 2,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "selected_model"
    ] == "drift"

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
            36.0,
        ]
    )

    assert (
        forecast[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_automatic_forecast_with_intervals():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "automatic_forecast_with_intervals",
        ],
        parameters_by_method={
            "automatic_forecast_with_intervals": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 2,
                "coverage": 0.8,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "selected_model"
    ] == "drift"

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
            36.0,
        ]
    )

    assert len(
        forecast[
            "intervals"
        ]
    ) == 2


def test_pipeline_executes_automatic_forecast_reliability():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "automatic_forecast_reliability",
        ],
        parameters_by_method={
            "automatic_forecast_reliability": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 1,
                "coverage": 0.9,
                "calibration_tolerance": 0.1,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    reliability = result.result[
        "result"
    ]

    assert reliability[
        "selected_model"
    ] == "drift"

    assert reliability[
        "empirical_coverage"
    ] == pytest.approx(
        1.0
    )

    assert reliability[
        "calibration_status"
    ] == "well_calibrated"


def test_pipeline_executes_weighted_forecast_ensemble():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "weighted_forecast_ensemble",
        ],
        parameters_by_method={
            "weighted_forecast_ensemble": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 2,
                "metric": "mae",
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    ensemble = result.result[
        "result"
    ]

    assert ensemble[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
            36.0,
        ]
    )

    assert ensemble[
        "weights_sum"
    ] == pytest.approx(
        1.0
    )


def test_pipeline_executes_adaptive_forecast_ensemble_backtest():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "adaptive_forecast_ensemble_backtest",
        ],
        parameters_by_method={
            "adaptive_forecast_ensemble_backtest": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 1,
                "metric": "mae",
                "minimum_weight_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    evaluation = result.result[
        "result"
    ]

    assert evaluation[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert (
        evaluation[
            "look_ahead_leakage"
        ]
        is False
    )


def test_pipeline_executes_forecast_strategy_selection():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "forecast_strategy_selection",
        ],
        parameters_by_method={
            "forecast_strategy_selection": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 1,
                "metric": "mae",
                "minimum_weight_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    selection = result.result[
        "result"
    ]

    assert selection[
        "best_strategy"
    ] == "single_model"

    assert selection[
        "best_model"
    ] == "drift"

    assert selection[
        "best_score"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_pipeline_executes_automatic_strategy_forecast():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "automatic_strategy_forecast",
        ],
        parameters_by_method={
            "automatic_strategy_forecast": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 2,
                "metric": "mae",
                "minimum_weight_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "selected_strategy"
    ] == "single_model"

    assert forecast[
        "selected_model"
    ] == "drift"

    assert forecast[
        "forecast_values"
    ] == pytest.approx(
        [
            38.0,
            40.0,
        ]
    )


def test_pipeline_executes_automatic_strategy_forecast_with_intervals():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "automatic_strategy_forecast_with_intervals",
        ],
        parameters_by_method={
            "automatic_strategy_forecast_with_intervals": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 2,
                "metric": "mae",
                "coverage": 0.9,
                "minimum_weight_observations": 3,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    forecast = result.result[
        "result"
    ]

    assert forecast[
        "selected_strategy"
    ] == "single_model"

    assert forecast[
        "selected_model"
    ] == "drift"

    assert len(
        forecast[
            "intervals"
        ]
    ) == 2


def test_pipeline_executes_automatic_strategy_forecast_reliability():
    values = [
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
    ]

    dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ),
            "value": values,
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "automatic_strategy_forecast_reliability",
        ],
        parameters_by_method={
            "automatic_strategy_forecast_reliability": {
                "models": [
                    "naive",
                    "drift",
                ],
                "initial_train_size": 5,
                "horizon": 1,
                "coverage": 0.9,
                "calibration_tolerance": 0.1,
                "minimum_weight_observations": 3,
                "minimum_calibration_observations": 3,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    reliability = result.result[
        "result"
    ]

    assert reliability[
        "selected_model"
    ] == "drift"

    assert reliability[
        "empirical_coverage"
    ] == pytest.approx(
        1.0
    )

    assert reliability[
        "calibration_status"
    ] == "well_calibrated"


def test_pipeline_executes_missing_value_diagnostic():
    dataframe = pd.DataFrame(
        {
            "value": [
                10.0,
                None,
                30.0,
                None,
                50.0,
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "missing_value_diagnostic",
        ],
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "missing_count"
    ] == 2

    assert diagnostic[
        "missing_rate"
    ] == pytest.approx(
        0.4
    )

    assert diagnostic[
        "severity"
    ] == "high"


def test_pipeline_executes_constant_column_diagnostic():
    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "constant_column_diagnostic",
        ],
        parameters_by_method={
            "constant_column_diagnostic": {
                "dominance_threshold": 0.90,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "dominant_value"
    ] == "A"

    assert diagnostic[
        "dominant_ratio"
    ] == pytest.approx(
        0.9
    )

    assert (
        diagnostic[
            "is_quasi_constant"
        ]
        is True
    )


def test_pipeline_executes_duplicate_row_diagnostic_with_missing_values():
    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "B",
                "C",
            ],
            "value": [
                None,
                None,
                20,
                30,
            ],
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "duplicate_row_diagnostic",
        ],
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "total_count"
    ] == 4

    assert diagnostic[
        "missing_cell_count"
    ] == 2

    assert diagnostic[
        "duplicate_row_count"
    ] == 2

    assert diagnostic[
        "excess_duplicate_count"
    ] == 1

    assert result.rows_dropped == 0


def test_pipeline_executes_non_finite_value_diagnostic():
    dataframe = pd.DataFrame(
        {
            "value": [
                10.0,
                float("inf"),
                None,
                float("-inf"),
                50.0,
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "non_finite_value_diagnostic",
        ],
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "total_count"
    ] == 5

    assert diagnostic[
        "missing_count"
    ] == 1

    assert diagnostic[
        "infinity_count"
    ] == 2

    assert diagnostic[
        "non_finite_count"
    ] == 3

    assert result.rows_dropped == 0


def test_pipeline_executes_constant_column_diagnostic():
    dataframe = pd.DataFrame(
        {
            "status": [
                "A",
                "A",
                "A",
                "A",
                "B",
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "constant_column_diagnostic",
        ],
        parameters_by_method={
            "constant_column_diagnostic": {
                "dominance_threshold": 0.8,
            }
        },
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "dominant_ratio"
    ] == pytest.approx(
        0.8
    )

    assert (
        diagnostic[
            "is_near_constant"
        ]
        is True
    )

    assert diagnostic[
        "variability_status"
    ] == "near_constant"


def test_pipeline_executes_outlier_diagnostic():
    dataframe = pd.DataFrame(
        {
            "sales": [
                10.0,
                11.0,
                12.0,
                13.0,
                100.0,
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "outlier_diagnostic",
        ],
    )

    result = report.execution.results[
        0
    ]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "outlier_count"
    ] == 1

    assert diagnostic[
        "outlier_positions"
    ] == [4]

    assert (
        diagnostic[
            "has_outliers"
        ]
        is True
    )


def test_pipeline_executes_cardinality_diagnostic():
    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "cardinality_diagnostic",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    diagnostic = result.result["result"]

    assert diagnostic["unique_count"] == 3
    assert diagnostic["non_missing_count"] == 6
    assert diagnostic["cardinality_status"] == "normal"


def test_pipeline_executes_type_consistency_diagnostic():
    dataframe = pd.DataFrame(
        {
            "category": [
                "A",
                "B",
                "C",
                "D",
            ]
        }
    )

    report = DataAnalysisPipeline().analyze_dataframe(
        dataframe,
        methods=[
            "type_consistency_diagnostic",
        ],
    )

    result = report.execution.results[0]

    assert result.status == "success"

    diagnostic = result.result[
        "result"
    ]

    assert diagnostic[
        "type_families"
    ] == ["string"]

    assert (
        diagnostic[
            "is_homogeneous"
        ]
        is True
    )

    assert diagnostic[
        "consistency_status"
    ] == "consistent"
