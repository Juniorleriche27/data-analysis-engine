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


def test_adaptive_ensemble_exact_drift_has_zero_error():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "adaptive_forecast_ensemble_backtest",
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
        "metrics"
    ]["mae"] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    for record in result[
        "records"
    ]:
        assert record[
            "model_weights"
        ]["drift"] == pytest.approx(
            1.0
        )


def test_adaptive_ensemble_weights_are_leakage_safe():
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
        "adaptive_forecast_ensemble_backtest",
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
        minimum_weight_observations=2,
    ).result

    for record in result[
        "records"
    ]:
        assert (
            record[
                "calibration_max_target_index"
            ]
            < record[
                "training_size"
            ]
        )

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )


def test_adaptive_ensemble_weights_sum_to_one():
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
        "adaptive_forecast_ensemble_backtest",
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

    for record in result[
        "records"
    ]:
        assert record[
            "weights_sum"
        ] == pytest.approx(
            1.0
        )


def test_adaptive_ensemble_reports_per_horizon_metrics():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "adaptive_forecast_ensemble_backtest",
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
        minimum_weight_observations=2,
    ).result

    assert len(
        result[
            "per_horizon"
        ]
    ) == 2

    assert result[
        "per_horizon"
    ][0][
        "forecast_step"
    ] == 1

    assert result[
        "per_horizon"
    ][1][
        "forecast_step"
    ] == 2


def test_adaptive_ensemble_preserves_model_parameters():
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
        "adaptive_forecast_ensemble_backtest",
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
        minimum_weight_observations=2,
    ).result

    assert result[
        "parameters_by_model"
    ][
        "autoregressive"
    ] == {
        "p": 1,
    }

    assert result[
        "parameters_by_model"
    ][
        "simple_exponential_smoothing"
    ] == {
        "alpha": 0.8,
    }


def test_adaptive_ensemble_rejects_insufficient_weight_history():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "adaptive_forecast_ensemble_backtest",
            time=pd.date_range(
                "2026-01-01",
                periods=7,
                freq="D",
            ).tolist(),
            values=[
                10,
                11,
                12,
                13,
                14,
                15,
                16,
            ],
            models=[
                "naive",
                "drift",
            ],
            initial_train_size=5,
            horizon=1,
            minimum_weight_observations=5,
        )
