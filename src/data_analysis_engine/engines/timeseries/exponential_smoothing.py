from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _future_timestamps,
    _prepare_regular_series,
    _validate_horizon,
)


def _validate_alpha(
    value: Any,
) -> float:
    if (
        not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
        or isinstance(
            value,
            (bool, np.bool_),
        )
    ):
        raise InvalidAnalysisInputError(
            "'alpha' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0.0
        or value > 1.0
    ):
        raise InvalidAnalysisInputError(
            "'alpha' must be greater than 0 "
            "and less than or equal to 1."
        )

    return value


def _simple_exponential_smoothing(
    values: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    if len(values) < 1:
        raise InvalidAnalysisInputError(
            "Simple exponential smoothing requires "
            "at least one observation."
        )

    level = float(
        values[0]
    )

    levels: list[float] = [
        level
    ]

    fitted_values: list[
        float | None
    ] = [
        None
    ]

    residuals: list[
        float | None
    ] = [
        None
    ]

    for index in range(
        1,
        len(values),
    ):
        fitted = float(
            level
        )

        observation = float(
            values[index]
        )

        residual = float(
            observation
            - fitted
        )

        level = float(
            alpha
            * observation
            + (
                1.0 - alpha
            )
            * level
        )

        fitted_values.append(
            fitted
        )

        residuals.append(
            residual
        )

        levels.append(
            level
        )

    return {
        "final_level": level,
        "levels": levels,
        "fitted_values": (
            fitted_values
        ),
        "residuals": residuals,
    }


def _simple_exponential_smoothing_forecast(
    values: np.ndarray,
    *,
    alpha: float,
    horizon: int,
) -> np.ndarray:
    fit = (
        _simple_exponential_smoothing(
            values,
            alpha=alpha,
        )
    )

    return np.full(
        horizon,
        fit[
            "final_level"
        ],
        dtype=float,
    )


class SimpleExponentialSmoothingForecast(
    AnalysisMethod
):
    name = (
        "simple_exponential_smoothing_forecast"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Forecast a non-seasonal time series using "
        "simple exponential smoothing with a fixed "
        "smoothing parameter alpha."
    )

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

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        alpha = _validate_alpha(
            kwargs.get(
                "alpha",
                0.2,
            )
        )

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

        fit = (
            _simple_exponential_smoothing(
                values,
                alpha=alpha,
            )
        )

        forecasts = np.full(
            horizon,
            fit[
                "final_level"
            ],
            dtype=float,
        )

        return {
            "model": (
                "simple_exponential_smoothing"
            ),
            "sample_size": len(
                values
            ),
            "alpha": alpha,
            "initial_level": float(
                values[0]
            ),
            "final_level": fit[
                "final_level"
            ],
            "horizon": horizon,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "levels": fit[
                "levels"
            ],
            "fitted_timestamps": [
                timestamp.isoformat()
                for timestamp in time
            ],
            "fitted_values": fit[
                "fitted_values"
            ],
            "residuals": fit[
                "residuals"
            ],
            "forecast_timestamps": (
                _future_timestamps(
                    time,
                    interval_ns,
                    horizon,
                )
            ),
            "forecast_values": [
                float(value)
                for value in forecasts
            ],
        }
