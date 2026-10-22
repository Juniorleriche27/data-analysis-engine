import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def build_exact_ar1_series():
    values = [10.0]

    for _ in range(7):
        values.append(
            2.0
            + 0.5
            * values[-1]
        )

    return values


def test_ar1_recovers_exact_coefficients():
    values = build_exact_ar1_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        p=1,
        horizon=1,
    )

    assert result.result[
        "intercept"
    ] == pytest.approx(
        2.0
    )

    assert result.result[
        "coefficients"
    ]["lag_1"] == pytest.approx(
        0.5
    )

    assert result.result[
        "in_sample_rmse"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_ar2_recovers_exact_coefficients():
    values = [
        10.0,
        8.0,
    ]

    for _ in range(10):
        values.append(
            1.0
            + 0.6
            * values[-1]
            - 0.2
            * values[-2]
        )

    result = DataAnalysisEngine().analyze(
        "autoregressive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        p=2,
        horizon=1,
    )

    assert result.result[
        "intercept"
    ] == pytest.approx(
        1.0
    )

    assert result.result[
        "coefficients"
    ]["lag_1"] == pytest.approx(
        0.6
    )

    assert result.result[
        "coefficients"
    ]["lag_2"] == pytest.approx(
        -0.2
    )


def test_ar_forecast_is_recursive():
    values = build_exact_ar1_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        p=1,
        horizon=3,
    )

    assert result.result[
        "forecast_values"
    ] == pytest.approx(
        [
            4.0234375,
            4.01171875,
            4.005859375,
        ]
    )


def test_ar_generates_future_timestamps():
    values = build_exact_ar1_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_forecast",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        p=1,
        horizon=2,
    )

    assert result.result[
        "forecast_timestamps"
    ] == [
        "2026-01-09T00:00:00+00:00",
        "2026-01-10T00:00:00+00:00",
    ]


def test_ar_rejects_invalid_order():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "autoregressive_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=5,
                freq="D",
            ).tolist(),
            values=[
                1,
                2,
                3,
                4,
                5,
            ],
            p=0,
        )


def test_ar_rejects_insufficient_observations():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "autoregressive_forecast",
            time=pd.date_range(
                "2026-01-01",
                periods=4,
                freq="D",
            ).tolist(),
            values=[
                1,
                2,
                3,
                4,
            ],
            p=2,
        )
