import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_seasonal_naive_repeats_last_cycle():
    result = DataAnalysisEngine().analyze(
        "seasonal_naive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=6,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
            10,
            20,
            30,
        ],
        season_length=3,
        horizon=4,
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            10,
            20,
            30,
            10,
        ]
    )


def test_seasonal_naive_generates_future_timestamps():
    result = DataAnalysisEngine().analyze(
        "seasonal_naive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=6,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
            10,
            20,
            30,
        ],
        season_length=3,
        horizon=2,
    )

    assert result.result[
        "forecast_timestamps"
    ] == [
        "2026-01-07T00:00:00+00:00",
        "2026-01-08T00:00:00+00:00",
    ]


def test_seasonal_naive_rejects_insufficient_history():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "seasonal_naive_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ).tolist(),
            values=[
                10,
                20,
                30,
                40,
            ],
            season_length=5,
            horizon=1,
        )


def test_seasonal_naive_backtest_is_exact_on_periodic_series():
    values = [
        10,
        20,
        30,
        10,
        20,
        30,
        10,
        20,
        30,
        10,
        20,
        30,
    ]

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="seasonal_naive",
        season_length=3,
        initial_train_size=6,
        horizon=1,
    )

    assert result.result[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0
    )

    assert result.result[
        "model_parameters"
    ] == {
        "season_length": 3,
    }


def test_seasonal_naive_backtest_supports_multistep():
    values = [
        10,
        20,
        30,
        10,
        20,
        30,
        10,
        20,
        30,
        10,
        20,
        30,
    ]

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="seasonal_naive",
        season_length=3,
        initial_train_size=6,
        horizon=3,
    )

    assert result.result[
        "metrics"
    ]["rmse"] == pytest.approx(
        0.0
    )


def test_seasonal_backtest_rejects_small_training_window():
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
                1,
                2,
                3,
                4,
                1,
                2,
            ],
            model="seasonal_naive",
            season_length=4,
            initial_train_size=3,
            horizon=1,
        )
