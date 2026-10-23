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


def test_model_selection_selects_best_model():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "forecast_model_selection",
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
        horizon=1,
        metric="mae",
    ).result

    assert result[
        "best_model"
    ] == "drift"

    assert result[
        "best_score"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_model_selection_uses_comparable_origins():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "forecast_model_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        models=[
            "naive",
            "drift",
            "autoregressive",
        ],
        parameters_by_model={
            "autoregressive": {
                "p": 1,
            }
        },
        initial_train_size=5,
        horizon=1,
    ).result

    origin_counts = {
        candidate[
            "origin_count"
        ]
        for candidate
        in result[
            "candidates"
        ]
    }

    assert len(
        origin_counts
    ) == 1

    assert (
        result[
            "comparable_origins"
        ]
        is True
    )

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )


def test_model_selection_supports_rmse():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "forecast_model_selection",
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
    ).result

    assert result[
        "metric"
    ] == "rmse"

    best = result[
        "candidates"
    ][0]

    assert result[
        "best_score"
    ] == pytest.approx(
        best[
            "metrics"
        ]["rmse"]
    )


def test_model_selection_preserves_model_parameters():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "forecast_model_selection",
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
        candidate[
            "model"
        ]: candidate[
            "model_parameters"
        ]
        for candidate
        in result[
            "candidates"
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


def test_model_selection_rejects_duplicate_models():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "forecast_model_selection",
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
        )


def test_model_selection_rejects_unsupported_model():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "forecast_model_selection",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            models=[
                "naive",
                "magic_forecast",
            ],
            initial_train_size=5,
        )
