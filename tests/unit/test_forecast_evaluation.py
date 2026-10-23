import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_naive_backtest_metrics():
    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=5,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            14,
            16,
            18,
        ],
        model="naive",
        initial_train_size=3,
        horizon=1,
    )

    metrics = result.result[
        "metrics"
    ]

    assert metrics[
        "mae"
    ] == pytest.approx(2.0)

    assert metrics[
        "rmse"
    ] == pytest.approx(2.0)

    assert result.result[
        "origin_count"
    ] == 2


def test_drift_backtest_is_exact_on_linear_series():
    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=6,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            14,
            16,
            18,
            20,
        ],
        model="drift",
        initial_train_size=3,
        horizon=1,
    )

    metrics = result.result[
        "metrics"
    ]

    assert metrics[
        "mae"
    ] == pytest.approx(0.0)

    assert metrics[
        "rmse"
    ] == pytest.approx(0.0)

    assert metrics[
        "smape"
    ] == pytest.approx(0.0)


def test_backtest_supports_multi_step_horizon():
    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=7,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            14,
            16,
            18,
            20,
            22,
        ],
        model="drift",
        initial_train_size=3,
        horizon=2,
    )

    assert result.result[
        "horizon"
    ] == 2

    assert result.result[
        "origin_count"
    ] == 3

    assert result.result[
        "forecast_observation_count"
    ] == 6

    assert result.result[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0
    )


def test_backtest_has_no_lookahead_leakage():
    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=5,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
            40,
            50,
        ],
        model="naive",
        initial_train_size=3,
    )

    first = result.result[
        "records"
    ][0]

    assert (
        result.result[
            "look_ahead_leakage"
        ]
        is False
    )

    assert first[
        "training_end_index"
    ] == 2

    assert first[
        "target_index"
    ] == 3

    assert (
        first[
            "training_end_index"
        ]
        < first[
            "target_index"
        ]
    )


def test_mape_ignores_zero_actual_values():
    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=5,
            freq="D",
        ).tolist(),
        values=[
            1,
            2,
            3,
            0,
            4,
        ],
        model="naive",
        initial_train_size=3,
        horizon=1,
    )

    metrics = result.result[
        "metrics"
    ]

    assert metrics[
        "mape_defined"
    ] is True

    assert metrics[
        "mape_observations"
    ] == 1


def test_backtest_rejects_unknown_model():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "rolling_origin_forecast_backtest",
            time=pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ).tolist(),
            values=[
                10,
                20,
                30,
                40,
                50,
            ],
            model="magic_model",
            initial_train_size=3,
        )


def _build_exact_backtest_ar1_series():
    values = [
        10.0
    ]

    for _ in range(11):
        values.append(
            2.0
            + 0.5
            * values[-1]
        )

    return values


def test_autoregressive_backtest_is_exact_on_ar1_series():
    values = _build_exact_backtest_ar1_series()

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="autoregressive",
        p=1,
        initial_train_size=5,
        horizon=1,
    )

    backtest = result.result

    assert backtest[
        "model"
    ] == "autoregressive"

    assert backtest[
        "model_parameters"
    ] == {
        "p": 1,
    }

    assert backtest[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert backtest[
        "metrics"
    ]["rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_autoregressive_backtest_supports_recursive_multistep():
    values = _build_exact_backtest_ar1_series()

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="autoregressive",
        p=1,
        initial_train_size=5,
        horizon=2,
    )

    backtest = result.result

    assert backtest[
        "horizon"
    ] == 2

    assert backtest[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_autoregressive_backtest_rejects_small_training_window():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "rolling_origin_forecast_backtest",
            time=pd.date_range(
                "2026-01-01",
                periods=10,
                freq="D",
            ).tolist(),
            values=[
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
            model="autoregressive",
            p=2,
            initial_train_size=4,
            horizon=1,
        )


def test_autoregressive_backtest_rejects_invalid_order():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "rolling_origin_forecast_backtest",
            time=pd.date_range(
                "2026-01-01",
                periods=8,
                freq="D",
            ).tolist(),
            values=[
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            model="autoregressive",
            p=0,
            initial_train_size=5,
        )


def test_autoregressive_backtest_preserves_no_lookahead_rule():
    values = _build_exact_backtest_ar1_series()

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="autoregressive",
        p=1,
        initial_train_size=5,
        horizon=1,
    )

    first = result.result[
        "records"
    ][0]

    assert (
        result.result[
            "look_ahead_leakage"
        ]
        is False
    )

    assert first[
        "training_end_index"
    ] == 4

    assert first[
        "target_index"
    ] == 5

    assert (
        first[
            "training_end_index"
        ]
        < first[
            "target_index"
        ]
    )
