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
