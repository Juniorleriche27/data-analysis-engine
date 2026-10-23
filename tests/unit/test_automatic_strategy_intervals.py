import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.automatic_strategy_forecast import (
    AutomaticStrategyForecast,
)
from data_analysis_engine.engines.timeseries.ensemble_backtest import (
    AdaptiveForecastEnsembleBacktest,
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


def test_strategy_intervals_select_drift():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_with_intervals",
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
        coverage=0.9,
        minimum_weight_observations=3,
        minimum_calibration_observations=3,
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


def test_strategy_intervals_exact_drift_zero_width():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_with_intervals",
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
        coverage=0.8,
        minimum_weight_observations=3,
        minimum_calibration_observations=3,
    ).result

    for interval in result[
        "intervals"
    ]:
        assert interval[
            "radius"
        ] == pytest.approx(
            0.0,
            abs=1e-12,
        )

        assert interval[
            "lower_bound"
        ] == pytest.approx(
            interval[
                "point_forecast"
            ]
        )

        assert interval[
            "upper_bound"
        ] == pytest.approx(
            interval[
                "point_forecast"
            ]
        )


def test_strategy_intervals_are_horizon_specific():
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
        47,
        49,
    ]

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_with_intervals",
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
        coverage=0.8,
        minimum_weight_observations=2,
        minimum_calibration_observations=3,
    ).result

    assert len(
        result[
            "intervals"
        ]
    ) == 2

    assert result[
        "intervals"
    ][0][
        "forecast_step"
    ] == 1

    assert result[
        "intervals"
    ][1][
        "forecast_step"
    ] == 2


def test_strategy_intervals_single_model_source():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_with_intervals",
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
        coverage=0.8,
        minimum_weight_observations=3,
        minimum_calibration_observations=3,
    ).result

    assert result[
        "calibration_source"
    ] == (
        "selected_single_model_backtest"
    )

    assert (
        result[
            "finite_sample_correction"
        ]
        is True
    )


def test_strategy_intervals_support_ensemble_branch(
    monkeypatch,
):
    values = _linear_series()

    def fake_automatic_execute(
        self,
        **kwargs,
    ):
        return {
            "selected_strategy": (
                "adaptive_ensemble"
            ),
            "selected_strategy_name": (
                "adaptive_ensemble"
            ),
            "selected_model": None,
            "selected_model_parameters": None,
            "strategy_selection_score": 1.0,
            "strategy_selection_metrics": {
                "mae": 1.0,
            },
            "strategy_ranking": [],
            "strategies_evaluated": 3,
            "models_evaluated": 2,
            "models": [
                "naive",
                "drift",
            ],
            "initial_train_size": 5,
            "common_evaluation_observations": 5,
            "ensemble_weights": {
                "naive": 0.25,
                "drift": 0.75,
            },
            "forecast_timestamps": (
                pd.date_range(
                    "2026-01-15",
                    periods=1,
                    freq="D",
                    tz="UTC",
                ).astype(
                    str
                ).tolist()
            ),
            "forecast_values": [
                38.0,
            ],
        }

    def fake_ensemble_backtest(
        self,
        **kwargs,
    ):
        return {
            "records": [
                {
                    "origin": 5,
                    "forecast_step": 1,
                    "target_index": 5,
                    "error": 1.0,
                },
                {
                    "origin": 6,
                    "forecast_step": 1,
                    "target_index": 6,
                    "error": -2.0,
                },
                {
                    "origin": 7,
                    "forecast_step": 1,
                    "target_index": 7,
                    "error": 1.5,
                },
            ],
        }

    monkeypatch.setattr(
        AutomaticStrategyForecast,
        "execute",
        fake_automatic_execute,
    )

    monkeypatch.setattr(
        AdaptiveForecastEnsembleBacktest,
        "execute",
        fake_ensemble_backtest,
    )

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_with_intervals",
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
        coverage=0.8,
        minimum_weight_observations=2,
        minimum_calibration_observations=3,
    ).result

    assert result[
        "selected_strategy"
    ] == "adaptive_ensemble"

    assert result[
        "calibration_source"
    ] == (
        "adaptive_ensemble_backtest"
    )

    assert result[
        "intervals"
    ][0][
        "radius"
    ] == pytest.approx(
        2.0
    )


def test_strategy_intervals_reject_invalid_coverage():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "automatic_strategy_forecast_with_intervals",
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
            coverage=1.0,
        )
