import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _linear_series():
    return [
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


def test_automatic_forecast_selects_drift_and_forecasts():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "naive",
            "drift",
            "simple_exponential_smoothing",
        ],
        parameters_by_model={
            "simple_exponential_smoothing": {
                "alpha": 0.5,
            }
        },
        initial_train_size=5,
        horizon=2,
        metric="mae",
    ).result

    assert result[
        "selected_model"
    ] == "drift"

    assert result[
        "selection_score"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
            36.0,
        ]
    )


def test_automatic_forecast_uses_full_history_after_selection():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "drift",
        ],
        initial_train_size=5,
        horizon=1,
    ).result

    assert (
        result[
            "refit_on_full_history"
        ]
        is True
    )

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
        ]
    )


def test_automatic_forecast_preserves_selected_parameters():
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

    result = DataAnalysisEngine().analyze(
        "automatic_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "holt_winters_additive",
        ],
        parameters_by_model={
            "holt_winters_additive": {
                "alpha": 0.2,
                "beta": 0.2,
                "gamma": 0.2,
                "season_length": 4,
            }
        },
        initial_train_size=8,
        horizon=2,
    ).result

    assert result[
        "selected_model_parameters"
    ] == {
        "alpha": 0.2,
        "beta": 0.2,
        "gamma": 0.2,
        "season_length": 4,
    }


def test_automatic_forecast_returns_model_ranking():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "naive",
            "drift",
        ],
        initial_train_size=5,
        horizon=1,
    ).result

    assert len(
        result[
            "model_ranking"
        ]
    ) == 2

    assert result[
        "model_ranking"
    ][0][
        "rank"
    ] == 1

    assert result[
        "model_ranking"
    ][0][
        "model"
    ] == "drift"


def test_automatic_forecast_is_leakage_safe():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "naive",
            "drift",
        ],
        initial_train_size=5,
        horizon=1,
    ).result

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )

    assert (
        result[
            "comparable_origins"
        ]
        is True
    )


def test_automatic_forecast_rejects_duplicate_models():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "automatic_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            models=[
                "naive",
                "naive",
            ],
            initial_train_size=5,
            horizon=1,
        )
