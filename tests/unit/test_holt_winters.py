import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _trend_seasonal_series():
    return [
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


def test_holt_winters_exact_additive_series():
    values = _trend_seasonal_series()

    result = DataAnalysisEngine().analyze(
        "holt_winters_additive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        season_length=4,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        horizon=4,
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            142.0,
            129.0,
            141.0,
            128.0,
        ]
    )


def test_holt_winters_initialization_recovers_trend():
    values = _trend_seasonal_series()

    result = DataAnalysisEngine().analyze(
        "holt_winters_additive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        season_length=4,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
    )

    assert result.result[
        "initial_trend"
    ] == pytest.approx(
        2.0
    )

    assert result.result[
        "initial_seasonals"
    ] == pytest.approx(
        [
            10.0,
            -5.0,
            5.0,
            -10.0,
        ]
    )


def test_holt_winters_rejects_less_than_two_seasons():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "holt_winters_additive_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=7,
                freq="D",
            ).tolist(),
            values=[
                10,
                20,
                30,
                40,
                11,
                21,
                31,
            ],
            season_length=4,
        )


def test_holt_winters_rejects_invalid_gamma():
    values = _trend_seasonal_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "holt_winters_additive_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            season_length=4,
            gamma=0.0,
        )


def test_holt_winters_backtest_is_exact():
    values = _trend_seasonal_series()

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="holt_winters_additive",
        season_length=4,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        initial_train_size=8,
        horizon=1,
    )

    assert result.result[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result.result[
        "metrics"
    ]["rmse"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_holt_winters_backtest_preserves_parameters():
    values = _trend_seasonal_series()

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="holt_winters_additive",
        season_length=4,
        alpha=0.8,
        beta=0.5,
        gamma=0.4,
        initial_train_size=8,
        horizon=1,
    )

    assert result.result[
        "model_parameters"
    ] == {
        "alpha": 0.8,
        "beta": 0.5,
        "gamma": 0.4,
        "season_length": 4,
    }

    assert (
        result.result[
            "look_ahead_leakage"
        ]
        is False
    )
