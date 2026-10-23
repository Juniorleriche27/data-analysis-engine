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


def test_ensemble_exact_model_receives_full_weight():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "weighted_forecast_ensemble",
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
    ).result

    weights = {
        item[
            "model"
        ]: item[
            "weight"
        ]
        for item
        in result[
            "component_forecasts"
        ]
    }

    assert weights[
        "drift"
    ] == pytest.approx(
        1.0
    )

    assert weights[
        "naive"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
            36.0,
        ]
    )


def test_ensemble_weights_sum_to_one():
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
    ]

    result = DataAnalysisEngine().analyze(
        "weighted_forecast_ensemble",
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
    ).result

    assert result[
        "weights_sum"
    ] == pytest.approx(
        1.0
    )

    assert sum(
        item[
            "weight"
        ]
        for item
        in result[
            "component_forecasts"
        ]
    ) == pytest.approx(
        1.0
    )


def test_ensemble_respects_top_k():
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
    ]

    result = DataAnalysisEngine().analyze(
        "weighted_forecast_ensemble",
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
        top_k=2,
    ).result

    assert result[
        "models_in_ensemble"
    ] == 2

    assert len(
        result[
            "component_forecasts"
        ]
    ) == 2


def test_ensemble_preserves_component_parameters():
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
    ]

    result = DataAnalysisEngine().analyze(
        "weighted_forecast_ensemble",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "autoregressive",
            "simple_exponential_smoothing",
        ],
        parameters_by_model={
            "autoregressive": {
                "p": 1,
            },
            "simple_exponential_smoothing": {
                "alpha": 0.8,
            },
        },
        initial_train_size=5,
        horizon=1,
    ).result

    parameters = {
        item[
            "model"
        ]: item[
            "model_parameters"
        ]
        for item
        in result[
            "component_forecasts"
        ]
    }

    assert parameters[
        "autoregressive"
    ] == {
        "p": 1,
    }

    assert parameters[
        "simple_exponential_smoothing"
    ] == {
        "alpha": 0.8,
    }


def test_ensemble_is_leakage_safe():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "weighted_forecast_ensemble",
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

    assert (
        result[
            "refit_components_on_full_history"
        ]
        is True
    )


def test_ensemble_rejects_top_k_above_model_count():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "weighted_forecast_ensemble",
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
            top_k=3,
        )
