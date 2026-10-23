import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_holt_exact_linear_series():
    result = DataAnalysisEngine().analyze(
        "holt_linear_trend_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            14,
            16,
        ],
        alpha=1.0,
        beta=1.0,
        horizon=2,
    )

    assert result.result[
        "final_level"
    ] == pytest.approx(
        16.0
    )

    assert result.result[
        "final_trend"
    ] == pytest.approx(
        2.0
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            18.0,
            20.0,
        ]
    )


def test_holt_tracks_level_and_trend():
    result = DataAnalysisEngine().analyze(
        "holt_linear_trend_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            14,
            16,
        ],
        alpha=0.5,
        beta=0.5,
    )

    assert len(
        result.result[
            "levels"
        ]
    ) == 4

    assert len(
        result.result[
            "trends"
        ]
    ) == 4


def test_holt_rejects_invalid_beta():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "holt_linear_trend_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ).tolist(),
            values=[
                10,
                12,
                14,
                16,
            ],
            alpha=0.5,
            beta=0.0,
        )


def test_holt_backtest_is_exact_on_linear_series():
    values = [
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
    ]

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="holt_linear_trend",
        alpha=1.0,
        beta=1.0,
        initial_train_size=4,
        horizon=1,
    )

    assert result.result[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0
    )

    assert result.result[
        "metrics"
    ]["rmse"] == pytest.approx(
        0.0
    )


def test_holt_backtest_supports_multistep():
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
    ]

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="holt_linear_trend",
        alpha=1.0,
        beta=1.0,
        initial_train_size=4,
        horizon=2,
    )

    assert result.result[
        "metrics"
    ]["mae"] == pytest.approx(
        0.0
    )


def test_holt_backtest_preserves_parameters_and_no_leakage():
    values = [
        10,
        12,
        14,
        16,
        18,
        20,
    ]

    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="holt_linear_trend",
        alpha=0.8,
        beta=0.4,
        initial_train_size=4,
        horizon=1,
    )

    assert result.result[
        "model_parameters"
    ] == {
        "alpha": 0.8,
        "beta": 0.4,
    }

    assert (
        result.result[
            "look_ahead_leakage"
        ]
        is False
    )
