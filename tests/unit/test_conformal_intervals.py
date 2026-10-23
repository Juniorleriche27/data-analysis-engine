from math import ceil

import numpy as np
import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_conformal_interval_zero_width_for_exact_drift():
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
        "conformal_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="drift",
        initial_train_size=5,
        horizon=1,
        coverage=0.8,
        minimum_calibration_observations=3,
    ).result

    interval = result[
        "intervals"
    ][0]

    assert interval[
        "radius"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert interval[
        "interval_width"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_conformal_interval_is_symmetric():
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

    result = DataAnalysisEngine().analyze(
        "conformal_forecast_interval",
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
        minimum_calibration_observations=3,
    ).result

    interval = result[
        "intervals"
    ][0]

    lower_distance = (
        interval[
            "point_forecast"
        ]
        - interval[
            "lower_bound"
        ]
    )

    upper_distance = (
        interval[
            "upper_bound"
        ]
        - interval[
            "point_forecast"
        ]
    )

    assert lower_distance == pytest.approx(
        upper_distance
    )


def test_conformal_interval_uses_finite_sample_rank():
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

    absolute_errors = np.sort(
        np.asarray(
            [
                abs(
                    record[
                        "error"
                    ]
                )
                for record
                in backtest[
                    "records"
                ]
            ],
            dtype=float,
        )
    )

    coverage = 0.8

    rank = min(
        ceil(
            (
                len(
                    absolute_errors
                )
                + 1
            )
            * coverage
        ),
        len(
            absolute_errors
        ),
    )

    expected_radius = (
        absolute_errors[
            rank - 1
        ]
    )

    result = engine.analyze(
        "conformal_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="naive",
        initial_train_size=5,
        horizon=1,
        coverage=coverage,
        minimum_calibration_observations=3,
    ).result

    interval = result[
        "intervals"
    ][0]

    assert interval[
        "conformal_rank"
    ] == rank

    assert interval[
        "radius"
    ] == pytest.approx(
        expected_radius
    )


def test_conformal_interval_supports_horizon_specific_calibration():
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
        "conformal_forecast_interval",
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


def test_conformal_interval_does_not_claim_iid_guarantee():
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
        "conformal_forecast_interval",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        model="drift",
        initial_train_size=5,
        horizon=1,
        minimum_calibration_observations=3,
    ).result

    assert (
        result[
            "iid_coverage_guarantee_claimed"
        ]
        is False
    )

    assert (
        result[
            "finite_sample_correction"
        ]
        is True
    )


def test_conformal_interval_rejects_insufficient_calibration():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "conformal_forecast_interval",
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
            minimum_calibration_observations=3,
        )
