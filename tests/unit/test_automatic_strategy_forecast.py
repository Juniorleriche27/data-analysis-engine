import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.forecast_ensemble import (
    WeightedForecastEnsemble,
)
from data_analysis_engine.engines.timeseries.strategy_selection import (
    ForecastStrategySelection,
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
        34,
        36,
    ]


def test_automatic_strategy_selects_drift_and_forecasts():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast",
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
        horizon=2,
        metric="mae",
        minimum_weight_observations=3,
    ).result

    assert result[
        "selected_strategy"
    ] == "single_model"

    assert result[
        "selected_model"
    ] == "drift"

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            38.0,
            40.0,
        ]
    )


def test_automatic_strategy_refits_single_model_on_full_history():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast",
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
        minimum_weight_observations=3,
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
            38.0,
        ]
    )


def test_automatic_strategy_preserves_fair_ranking():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast",
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
        minimum_weight_observations=3,
    ).result

    counts = {
        candidate[
            "evaluation_observations"
        ]
        for candidate
        in result[
            "strategy_ranking"
        ]
    }

    assert len(
        counts
    ) == 1

    assert (
        result[
            "same_evaluation_window"
        ]
        is True
    )


def test_automatic_strategy_executes_ensemble_branch(
    monkeypatch,
):
    values = _linear_series()

    def fake_selection_execute(
        self,
        **kwargs,
    ):
        return {
            "best_strategy": (
                "adaptive_ensemble"
            ),
            "best_strategy_name": (
                "adaptive_ensemble"
            ),
            "best_model": None,
            "best_model_parameters": None,
            "best_score": 1.0,
            "best_metrics": {
                "mae": 1.0,
            },
            "candidates": [
                {
                    "strategy": (
                        "single_model"
                    ),
                    "model": "naive",
                    "model_parameters": {},
                    "evaluation_observations": 5,
                },
                {
                    "strategy": (
                        "single_model"
                    ),
                    "model": "drift",
                    "model_parameters": {},
                    "evaluation_observations": 5,
                },
                {
                    "strategy": (
                        "adaptive_ensemble"
                    ),
                    "model": None,
                    "model_parameters": None,
                    "evaluation_observations": 5,
                },
            ],
            "strategies_evaluated": 3,
            "models_evaluated": 2,
            "models": [
                "naive",
                "drift",
            ],
            "initial_train_size": 5,
            "common_evaluation_observations": 5,
        }

    def fake_ensemble_execute(
        self,
        **kwargs,
    ):
        return {
            "forecast_values": [
                37.5,
                39.5,
            ],
            "component_forecasts": [
                {
                    "model": "naive",
                    "weight": 0.25,
                },
                {
                    "model": "drift",
                    "weight": 0.75,
                },
            ],
            "weights_sum": 1.0,
            "weighting_method": (
                "inverse_out_of_sample_error"
            ),
        }

    monkeypatch.setattr(
        ForecastStrategySelection,
        "execute",
        fake_selection_execute,
    )

    monkeypatch.setattr(
        WeightedForecastEnsemble,
        "execute",
        fake_ensemble_execute,
    )

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast",
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
        horizon=2,
    ).result

    assert result[
        "selected_strategy"
    ] == "adaptive_ensemble"

    assert result[
        "selected_model"
    ] is None

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            37.5,
            39.5,
        ]
    )

    assert result[
        "ensemble_weights"
    ] == {
        "naive": 0.25,
        "drift": 0.75,
    }


def test_automatic_strategy_is_leakage_safe():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast",
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
        minimum_weight_observations=3,
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


def test_automatic_strategy_requires_multiple_models():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "automatic_strategy_forecast",
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
        )
