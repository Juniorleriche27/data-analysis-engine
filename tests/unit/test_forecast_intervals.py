import numpy as np
import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_empirical_interval_is_zero_width_for_exact_drift():
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
        28,
    ]

    result = DataAnalysisEngine().analyze(
        "empirical_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="drift",
        initial_train_size=5,
        horizon=2,
        coverage=0.8,
        minimum_interval_observations=3,
    ).result

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            30.0,
            32.0,
        ]
    )

    for interval in result[
        "intervals"
    ]:
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


def test_empirical_interval_uses_backtest_error_quantiles():
    values = [
        10,
        12,
        9,
        13,
        11,
        14,
        12,
        15,
        13,
        16,
    ]

    engine = DataAnalysisEngine()

    backtest = engine.analyze(
        "rolling_origin_forecast_backtest",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="naive",
        initial_train_size=5,
        horizon=1,
    ).result

    errors = np.asarray(
        [
            record["error"]
            for record in backtest[
                "records"
            ]
        ],
        dtype=float,
    )

    result = engine.analyze(
        "empirical_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="naive",
        initial_train_size=5,
        horizon=1,
        coverage=0.8,
        minimum_interval_observations=3,
    ).result

    interval = result[
        "intervals"
    ][0]

    assert interval[
        "lower_error_quantile"
    ] == pytest.approx(
        np.quantile(
            errors,
            0.1,
        )
    )

    assert interval[
        "upper_error_quantile"
    ] == pytest.approx(
        np.quantile(
            errors,
            0.9,
        )
    )


def test_empirical_interval_supports_multiple_horizons():
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
        28,
    ]

    result = DataAnalysisEngine().analyze(
        "empirical_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="drift",
        initial_train_size=5,
        horizon=2,
        coverage=0.8,
        minimum_interval_observations=3,
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


def test_empirical_interval_preserves_model_parameters():
    values = [
        100,
        103,
        106,
        110,
        113,
        117,
        120,
        124,
        128,
        131,
    ]

    result = DataAnalysisEngine().analyze(
        "empirical_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="autoregressive",
        p=1,
        initial_train_size=5,
        horizon=1,
        coverage=0.8,
        minimum_interval_observations=3,
    ).result

    assert result[
        "model_parameters"
    ] == {
        "p": 1,
    }

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )


def test_empirical_interval_rejects_invalid_coverage():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "empirical_forecast_interval",
            time=pd.date_range(
                "2026-01-01",
                periods=8,
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
                17,
            ],
            model="naive",
            initial_train_size=4,
            coverage=1.0,
        )


def test_empirical_interval_rejects_insufficient_calibration_errors():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "empirical_forecast_interval",
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
            model="naive",
            initial_train_size=5,
            horizon=1,
            minimum_interval_observations=3,
        )
