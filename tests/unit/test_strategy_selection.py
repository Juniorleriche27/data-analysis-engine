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
        34,
        36,
    ]


def test_strategy_selection_prefers_exact_single_model_on_tie():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "forecast_strategy_selection",
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
        metric="mae",
        minimum_weight_observations=3,
    ).result

    assert result[
        "best_strategy"
    ] == "single_model"

    assert result[
        "best_model"
    ] == "drift"

    assert result[
        "best_score"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_strategy_selection_uses_same_evaluation_window():
    values = [
        10,
        13,
        14,
        18,
        19,
        23,
        24,
        28,
        29,
        34,
        35,
        39,
        41,
        44,
    ]

    result = DataAnalysisEngine().analyze(
        "forecast_strategy_selection",
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
                "alpha": 0.8,
            }
        },
        initial_train_size=5,
        horizon=1,
        minimum_weight_observations=2,
    ).result

    counts = {
        candidate[
            "evaluation_observations"
        ]
        for candidate
        in result[
            "candidates"
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


def test_strategy_selection_includes_adaptive_ensemble():
    values = [
        10,
        13,
        14,
        18,
        19,
        23,
        24,
        28,
        29,
        34,
        35,
        39,
        41,
        44,
    ]

    result = DataAnalysisEngine().analyze(
        "forecast_strategy_selection",
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
        minimum_weight_observations=2,
    ).result

    strategies = {
        candidate[
            "strategy"
        ]
        for candidate
        in result[
            "candidates"
        ]
    }

    assert (
        "single_model"
        in strategies
    )

    assert (
        "adaptive_ensemble"
        in strategies
    )


def test_strategy_selection_supports_rmse():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "forecast_strategy_selection",
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
        metric="rmse",
        minimum_weight_observations=2,
    ).result

    assert result[
        "metric"
    ] == "rmse"

    assert result[
        "best_score"
    ] == pytest.approx(
        result[
            "best_metrics"
        ]["rmse"]
    )


def test_strategy_selection_is_leakage_safe():
    values = [
        10,
        13,
        14,
        18,
        19,
        23,
        24,
        28,
        29,
        34,
        35,
        39,
        41,
        44,
    ]

    result = DataAnalysisEngine().analyze(
        "forecast_strategy_selection",
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
        minimum_weight_observations=2,
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


def test_strategy_selection_requires_multiple_models():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "forecast_strategy_selection",
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
