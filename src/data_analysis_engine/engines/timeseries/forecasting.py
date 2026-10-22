from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.basic import (
    _validate_time_series,
)


def _validate_horizon(
    value: Any,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise InvalidAnalysisInputError(
            "'horizon' must be an integer."
        )

    if value < 1:
        raise InvalidAnalysisInputError(
            "'horizon' must be at least 1."
        )

    return value


def _prepare_regular_series(
    time: Any,
    values: Any,
    *,
    tolerance_ratio: float = 0.05,
) -> tuple[
    pd.DatetimeIndex,
    np.ndarray,
    int,
]:
    time_values, numeric_values = _validate_time_series(
        time,
        values,
    )

    parsed = pd.to_datetime(
        time_values,
        errors="coerce",
        utc=True,
    )

    if parsed.isna().any():
        raise InvalidAnalysisInputError(
            "All timestamps must be valid."
        )

    time_ns = np.asarray(
        [
            timestamp.value
            for timestamp in parsed
        ],
        dtype=np.int64,
    )

    deltas = np.diff(
        time_ns
    )

    if np.any(
        deltas <= 0
    ):
        raise InvalidAnalysisInputError(
            "Forecasting requires strictly increasing "
            "unique timestamps."
        )

    expected_interval_ns = int(
        round(
            float(
                np.median(
                    deltas
                )
            )
        )
    )

    if expected_interval_ns <= 0:
        raise InvalidAnalysisInputError(
            "Unable to infer a positive temporal interval."
        )

    relative_deviations = (
        np.abs(
            deltas
            - expected_interval_ns
        )
        / expected_interval_ns
    )

    if np.any(
        relative_deviations
        > tolerance_ratio
    ):
        raise InvalidAnalysisInputError(
            "Forecasting requires a sufficiently regular "
            "time axis. Diagnose or resample the timeline first."
        )

    return (
        parsed,
        numeric_values,
        expected_interval_ns,
    )


def _future_timestamps(
    parsed: pd.DatetimeIndex,
    interval_ns: int,
    horizon: int,
) -> list[str]:
    last_ns = parsed[-1].value

    future_ns = [
        last_ns
        + interval_ns * step
        for step in range(
            1,
            horizon + 1,
        )
    ]

    future = pd.to_datetime(
        future_ns,
        unit="ns",
        utc=True,
    )

    return [
        timestamp.isoformat()
        for timestamp in future
    ]


class _ForecastMethod(AnalysisMethod):
    category = "timeseries"

    supported_roles = (
        "time",
        "measure",
    )

    minimum_compatible_columns = 2

    selection_mode = "time_series"
    analysis_unit = "series"

    role_requirements = (
        ("time", 1),
        ("measure", 1),
    )

    allow_constant_columns = True


class NaiveForecast(_ForecastMethod):
    name = "naive_forecast"
    version = "1.0"

    description = (
        "Forecast future observations using the final "
        "observed value as a persistence baseline."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        horizon = _validate_horizon(
            kwargs.get(
                "horizon",
                1,
            )
        )

        time, values, interval_ns = (
            _prepare_regular_series(
                kwargs.get("time"),
                kwargs.get("values"),
            )
        )

        last_value = float(
            values[-1]
        )

        forecasts = [
            last_value
            for _ in range(
                horizon
            )
        ]

        return {
            "model": "naive",
            "sample_size": len(values),
            "horizon": horizon,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "last_observed_value": (
                last_value
            ),
            "forecast_timestamps": (
                _future_timestamps(
                    time,
                    interval_ns,
                    horizon,
                )
            ),
            "forecast_values": forecasts,
        }


class DriftForecast(_ForecastMethod):
    name = "drift_forecast"
    version = "1.0"

    description = (
        "Forecast future observations by extending the "
        "average drift between the first and last observation."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        horizon = _validate_horizon(
            kwargs.get(
                "horizon",
                1,
            )
        )

        time, values, interval_ns = (
            _prepare_regular_series(
                kwargs.get("time"),
                kwargs.get("values"),
            )
        )

        if len(values) < 2:
            raise InvalidAnalysisInputError(
                "Drift forecasting requires at least "
                "two observations."
            )

        drift_per_period = float(
            (
                values[-1]
                - values[0]
            )
            / (
                len(values)
                - 1
            )
        )

        last_value = float(
            values[-1]
        )

        forecasts = [
            float(
                last_value
                + drift_per_period
                * step
            )
            for step in range(
                1,
                horizon + 1,
            )
        ]

        return {
            "model": "drift",
            "sample_size": len(values),
            "horizon": horizon,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "last_observed_value": (
                last_value
            ),
            "drift_per_period": (
                drift_per_period
            ),
            "forecast_timestamps": (
                _future_timestamps(
                    time,
                    interval_ns,
                    horizon,
                )
            ),
            "forecast_values": forecasts,
        }
