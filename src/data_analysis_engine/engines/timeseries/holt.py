from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.exponential_smoothing import (
    _validate_alpha,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _future_timestamps,
    _prepare_regular_series,
    _validate_horizon,
)


def _validate_beta(
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
            "'beta' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0.0
        or value > 1.0
    ):
        raise InvalidAnalysisInputError(
            "'beta' must be greater than 0 "
            "and less than or equal to 1."
        )

    return value


def _holt_linear_fit(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    if len(values) < 2:
        raise InvalidAnalysisInputError(
            "Holt linear trend requires at least "
            "two observations."
        )

    level = float(
        values[0]
    )

    trend = float(
        values[1]
        - values[0]
    )

    initial_level = level
    initial_trend = trend

    levels: list[float] = [
        level
    ]

    trends: list[float] = [
        trend
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
        previous_level = level
        previous_trend = trend

        fitted = float(
            previous_level
            + previous_trend
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
            * (
                previous_level
                + previous_trend
            )
        )

        trend = float(
            beta
            * (
                level
                - previous_level
            )
            + (
                1.0 - beta
            )
            * previous_trend
        )

        levels.append(
            level
        )

        trends.append(
            trend
        )

        fitted_values.append(
            fitted
        )

        residuals.append(
            residual
        )

    return {
        "initial_level": (
            initial_level
        ),
        "initial_trend": (
            initial_trend
        ),
        "final_level": (
            level
        ),
        "final_trend": (
            trend
        ),
        "levels": levels,
        "trends": trends,
        "fitted_values": (
            fitted_values
        ),
        "residuals": (
            residuals
        ),
    }


def _holt_linear_forecast(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
    horizon: int,
) -> np.ndarray:
    fit = _holt_linear_fit(
        values,
        alpha=alpha,
        beta=beta,
    )

    level = fit[
        "final_level"
    ]

    trend = fit[
        "final_trend"
    ]

    return np.asarray(
        [
            float(
                level
                + step
                * trend
            )
            for step in range(
                1,
                horizon + 1,
            )
        ],
        dtype=float,
    )


class HoltLinearTrendForecast(
    AnalysisMethod
):
    name = "holt_linear_trend_forecast"
    category = "timeseries"
    version = "1.0"

    description = (
        "Forecast a non-seasonal time series with "
        "Holt's linear trend method using level and "
        "trend smoothing parameters."
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

        beta = _validate_beta(
            kwargs.get(
                "beta",
                0.1,
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

        fit = _holt_linear_fit(
            values,
            alpha=alpha,
            beta=beta,
        )

        forecasts = (
            _holt_linear_forecast(
                values,
                alpha=alpha,
                beta=beta,
                horizon=horizon,
            )
        )

        return {
            "model": (
                "holt_linear_trend"
            ),
            "sample_size": len(
                values
            ),
            "alpha": alpha,
            "beta": beta,
            "initial_level": fit[
                "initial_level"
            ],
            "initial_trend": fit[
                "initial_trend"
            ],
            "final_level": fit[
                "final_level"
            ],
            "final_trend": fit[
                "final_trend"
            ],
            "horizon": horizon,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "levels": fit[
                "levels"
            ],
            "trends": fit[
                "trends"
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
