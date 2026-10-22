import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_naive_forecast_repeats_last_value():
    result = DataAnalysisEngine().analyze(
        "naive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            15,
            20,
        ],
        horizon=3,
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            20,
            20,
            20,
        ]
    )


def test_naive_forecast_generates_future_timestamps():
    result = DataAnalysisEngine().analyze(
        "naive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=3,
            freq="D",
        ).tolist(),
        values=[
            10,
            20,
            30,
        ],
        horizon=2,
    )

    assert result.result[
        "frequency_seconds"
    ] == pytest.approx(
        86400.0
    )

    assert result.result[
        "forecast_timestamps"
    ] == [
        "2026-01-04T00:00:00+00:00",
        "2026-01-05T00:00:00+00:00",
    ]


def test_drift_forecast_extends_average_drift():
    result = DataAnalysisEngine().analyze(
        "drift_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=4,
            freq="D",
        ).tolist(),
        values=[
            10,
            12,
            14,
            16,
        ],
        horizon=3,
    )

    assert result.result[
        "drift_per_period"
    ] == pytest.approx(
        2.0
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            18,
            20,
            22,
        ]
    )


def test_forecast_rejects_invalid_horizon():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "naive_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ).tolist(),
            values=[
                10,
                20,
                30,
            ],
            horizon=0,
        )


def test_forecast_rejects_irregular_time_axis():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "drift_forecast",
            time=pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-04",
                    "2026-01-05",
                ]
            ).tolist(),
            values=[
                10,
                20,
                30,
                40,
            ],
            horizon=2,
        )
