import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_ses_alpha_one_equals_last_observation():
    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
        ],
        alpha=1.0,
        horizon=1,
    )

    assert result.result[
        "final_level"
    ] == pytest.approx(
        30.0
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            30.0,
        ]
    )


def test_ses_level_recursion():
    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
        ],
        alpha=0.5,
        horizon=1,
    )

    assert result.result[
        "levels"
    ] == pytest.approx(
        [
            10.0,
            15.0,
            22.5,
        ]
    )

    assert result.result[
        "final_level"
    ] == pytest.approx(
        22.5
    )


def test_ses_multistep_forecast_is_flat():
    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
        ],
        alpha=0.5,
        horizon=3,
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            22.5,
            22.5,
            22.5,
        ]
    )


def test_ses_rejects_invalid_alpha():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "simple_exponential_smoothing_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ).tolist(),
            values=[
                10,
                20,
                30,
            ],
            alpha=0.0,
        )


def test_ses_backtest_alpha_one_matches_naive():
    time = pd.date_range(
        "2026-01-01",
        periods=8,
        freq="D",
    ).tolist()

    values = [
        10,
        12,
        11,
        14,
        13,
        16,
        15,
        18,
    ]

    engine = DataAnalysisEngine()

    naive = engine.analyze(
        "rolling_origin_forecast_backtest",
        time=time,
        values=values,
        model="naive",
        initial_train_size=4,
        horizon=1,
    ).result

    ses = engine.analyze(
        "rolling_origin_forecast_backtest",
        time=time,
        values=values,
        model="simple_exponential_smoothing",
        alpha=1.0,
        initial_train_size=4,
        horizon=1,
    ).result

    assert ses[
        "metrics"
    ]["mae"] == pytest.approx(
        naive[
            "metrics"
        ]["mae"]
    )

    assert ses[
        "metrics"
    ]["rmse"] == pytest.approx(
        naive[
            "metrics"
        ]["rmse"]
    )


def test_ses_backtest_preserves_alpha_parameter():
    result = DataAnalysisEngine().analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=8,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            11,
            14,
            13,
            16,
            15,
            18,
        ],
        model="simple_exponential_smoothing",
        alpha=0.4,
        initial_train_size=4,
        horizon=1,
    )

    assert result.result[
        "model_parameters"
    ] == {
        "alpha": 0.4,
    }

    assert (
        result.result[
            "look_ahead_leakage"
        ]
        is False
    )
