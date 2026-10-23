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


def test_automatic_interval_selects_drift_and_forecasts():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast_with_intervals",
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
        minimum_calibration_observations=3,
    ).result

    assert result[
        "selected_model"
    ] == "drift"

    assert result[
        "forecast_values"
    ] == pytest.approx(
        [
            34.0,
            36.0,
        ]
    )


def test_automatic_interval_exact_drift_has_zero_width():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast_with_intervals",
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
        horizon=2,
        coverage=0.8,
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


def test_automatic_interval_preserves_selected_parameters():
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
        "automatic_forecast_with_intervals",
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
        horizon=1,
        coverage=0.8,
        minimum_calibration_observations=3,
    ).result

    assert result[
        "selected_model_parameters"
    ] == {
        "alpha": 0.2,
        "beta": 0.2,
        "gamma": 0.2,
        "season_length": 4,
    }


def test_automatic_interval_point_forecasts_are_consistent():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast_with_intervals",
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
        minimum_calibration_observations=3,
    ).result

    for index, interval in enumerate(
        result[
            "intervals"
        ]
    ):
        assert interval[
            "point_forecast"
        ] == pytest.approx(
            result[
                "forecast_values"
            ][index]
        )


def test_automatic_interval_reports_safety_metadata():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_forecast_with_intervals",
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
        coverage=0.8,
        minimum_calibration_observations=3,
    ).result

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )

    assert (
        result[
            "finite_sample_correction"
        ]
        is True
    )

    assert (
        result[
            "iid_coverage_guarantee_claimed"
        ]
        is False
    )


def test_automatic_interval_rejects_invalid_coverage():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "automatic_forecast_with_intervals",
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
            coverage=1.0,
        )
