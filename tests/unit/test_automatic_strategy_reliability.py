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


def test_strategy_reliability_selects_drift():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_reliability",
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


def test_strategy_reliability_exact_drift_has_full_coverage():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_reliability",
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
        coverage=0.9,
        calibration_tolerance=0.1,
        minimum_weight_observations=3,
        minimum_calibration_observations=3,
    ).result

    assert result[
        "empirical_coverage"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "calibration_status"
    ] == "well_calibrated"

    assert (
        result[
            "reliable_within_tolerance"
        ]
        is True
    )


def test_strategy_reliability_is_sequentially_leakage_safe():
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
        "automatic_strategy_forecast_reliability",
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
        minimum_calibration_observations=2,
    ).result

    for record in result[
        "historical_calibration_records"
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
            "interval_calibration_look_ahead_leakage"
        ]
        is False
    )


def test_strategy_reliability_reports_per_horizon():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_reliability",
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
        minimum_calibration_observations=2,
    ).result

    assert len(
        result[
            "per_horizon_calibration"
        ]
    ) == 2

    assert result[
        "per_horizon_calibration"
    ][0][
        "forecast_step"
    ] == 1

    assert result[
        "per_horizon_calibration"
    ][1][
        "forecast_step"
    ] == 2


def test_strategy_reliability_reports_post_selection_limitation():
    values = _linear_series()

    result = DataAnalysisEngine().analyze(
        "automatic_strategy_forecast_reliability",
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
        minimum_calibration_observations=2,
    ).result

    assert (
        result[
            "post_selection_diagnostic"
        ]
        is True
    )

    assert (
        result[
            "selection_adjusted_coverage_guarantee"
        ]
        is False
    )

    assert (
        result[
            "iid_coverage_guarantee_claimed"
        ]
        is False
    )


def test_strategy_reliability_rejects_invalid_tolerance():
    values = _linear_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "automatic_strategy_forecast_reliability",
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
            calibration_tolerance=1.0,
        )
