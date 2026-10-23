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
    _validate_season_length,
)
from data_analysis_engine.engines.timeseries.holt import (
    _validate_beta,
)


def _validate_gamma(
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
            "'gamma' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0.0
        or value > 1.0
    ):
        raise InvalidAnalysisInputError(
            "'gamma' must be greater than 0 "
            "and less than or equal to 1."
        )

    return value


def _initialize_additive_holt_winters(
    values: np.ndarray,
    *,
    season_length: int,
) -> tuple[
    float,
    float,
    list[float],
]:
    minimum_sample_size = (
        2 * season_length
    )

    if len(values) < minimum_sample_size:
        raise InvalidAnalysisInputError(
            "Additive Holt-Winters requires at least "
            "two complete seasons."
        )

    first_season = values[
        :season_length
    ]

    second_season = values[
        season_length:
        2 * season_length
    ]

    trend = float(
        np.mean(
            second_season
            - first_season
        )
        / season_length
    )

    first_season_mean = float(
        np.mean(
            first_season
        )
    )

    level = float(
        first_season_mean
        + trend
        * (
            season_length - 1
        )
        / 2.0
    )

    seasonal_components: list[
        float
    ] = []

    for index in range(
        season_length
    ):
        baseline = float(
            level
            - trend
            * (
                season_length
                - 1
                - index
            )
        )

        seasonal_components.append(
            float(
                first_season[index]
                - baseline
            )
        )

    return (
        level,
        trend,
        seasonal_components,
    )


def _holt_winters_additive_fit(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    season_length: int,
) -> dict[str, Any]:
    (
        level,
        trend,
        initial_seasonals,
    ) = _initialize_additive_holt_winters(
        values,
        season_length=season_length,
    )

    initial_level = level
    initial_trend = trend

    seasonal_components = list(
        initial_seasonals
    )

    levels: list[
        float | None
    ] = [
        None
        for _ in range(
            season_length - 1
        )
    ] + [
        level
    ]

    trends: list[
        float | None
    ] = [
        None
        for _ in range(
            season_length - 1
        )
    ] + [
        trend
    ]

    fitted_values: list[
        float | None
    ] = [
        None
        for _ in range(
            season_length
        )
    ]

    residuals: list[
        float | None
    ] = [
        None
        for _ in range(
            season_length
        )
    ]

    for index in range(
        season_length,
        len(values),
    ):
        previous_level = level
        previous_trend = trend

        seasonal = float(
            seasonal_components[
                index
                - season_length
            ]
        )

        fitted = float(
            previous_level
            + previous_trend
            + seasonal
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
            * (
                observation
                - seasonal
            )
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

        new_seasonal = float(
            gamma
            * (
                observation
                - level
            )
            + (
                1.0 - gamma
            )
            * seasonal
        )

        seasonal_components.append(
            new_seasonal
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
        "initial_seasonals": (
            initial_seasonals
        ),
        "final_level": level,
        "final_trend": trend,
        "levels": levels,
        "trends": trends,
        "seasonal_components": (
            seasonal_components
        ),
        "fitted_values": (
            fitted_values
        ),
        "residuals": (
            residuals
        ),
    }


def _holt_winters_additive_forecast(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    season_length: int,
    horizon: int,
) -> np.ndarray:
    fit = (
        _holt_winters_additive_fit(
            values,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            season_length=season_length,
        )
    )

    level = float(
        fit[
            "final_level"
        ]
    )

    trend = float(
        fit[
            "final_trend"
        ]
    )

    latest_season = fit[
        "seasonal_components"
    ][
        -season_length:
    ]

    forecasts = []

    for step in range(
        1,
        horizon + 1,
    ):
        seasonal_index = (
            step - 1
        ) % season_length

        forecasts.append(
            float(
                level
                + step
                * trend
                + latest_season[
                    seasonal_index
                ]
            )
        )

    return np.asarray(
        forecasts,
        dtype=float,
    )


class HoltWintersAdditiveForecast(
    AnalysisMethod
):
    name = (
        "holt_winters_additive_forecast"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Forecast a regular time series using additive "
        "Holt-Winters level, trend, and seasonal components."
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

        gamma = _validate_gamma(
            kwargs.get(
                "gamma",
                0.1,
            )
        )

        season_length = (
            _validate_season_length(
                kwargs.get(
                    "season_length",
                    7,
                )
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
            _holt_winters_additive_fit(
                values,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                season_length=season_length,
            )
        )

        forecasts = (
            _holt_winters_additive_forecast(
                values,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                season_length=season_length,
                horizon=horizon,
            )
        )

        return {
            "model": (
                "holt_winters_additive"
            ),
            "sample_size": len(
                values
            ),
            "season_length": (
                season_length
            ),
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "initial_level": fit[
                "initial_level"
            ],
            "initial_trend": fit[
                "initial_trend"
            ],
            "initial_seasonals": fit[
                "initial_seasonals"
            ],
            "final_level": fit[
                "final_level"
            ],
            "final_trend": fit[
                "final_trend"
            ],
            "levels": fit[
                "levels"
            ],
            "trends": fit[
                "trends"
            ],
            "seasonal_components": fit[
                "seasonal_components"
            ],
            "fitted_values": fit[
                "fitted_values"
            ],
            "residuals": fit[
                "residuals"
            ],
            "horizon": horizon,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
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
