import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_conformal_calibration_exact_drift_has_full_coverage():
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
        30,
        32,
    ]

    result = DataAnalysisEngine().analyze(
        "conformal_interval_calibration",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="drift",
        initial_train_size=5,
        horizon=1,
        coverage=0.9,
        minimum_calibration_observations=3,
    ).result

    assert result[
        "empirical_coverage"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "average_interval_width"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_conformal_calibration_is_leakage_safe():
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
        14,
        17,
        15,
        18,
    ]

    result = DataAnalysisEngine().analyze(
        "conformal_interval_calibration",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="naive",
        initial_train_size=5,
        horizon=2,
        coverage=0.8,
        minimum_calibration_observations=2,
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


def test_conformal_calibration_reports_per_horizon():
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
        14,
        17,
        15,
        18,
    ]

    result = DataAnalysisEngine().analyze(
        "conformal_interval_calibration",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="naive",
        initial_train_size=5,
        horizon=2,
        coverage=0.8,
        minimum_calibration_observations=2,
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


def test_conformal_calibration_preserves_parameters():
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
        135,
        139,
    ]

    result = DataAnalysisEngine().analyze(
        "conformal_interval_calibration",
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
        minimum_calibration_observations=2,
    ).result

    assert result[
        "model_parameters"
    ] == {
        "p": 1,
    }


def test_conformal_calibration_reports_finite_sample_method():
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
        14,
        17,
    ]

    result = DataAnalysisEngine().analyze(
        "conformal_interval_calibration",
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
        minimum_calibration_observations=2,
    ).result

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


def test_conformal_calibration_rejects_insufficient_history():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "conformal_interval_calibration",
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
            minimum_calibration_observations=5,
        )
