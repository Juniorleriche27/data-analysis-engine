from math import sqrt
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


def _validate_order(
    value: Any,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise InvalidAnalysisInputError(
            "'p' must be an integer."
        )

    if value < 1:
        raise InvalidAnalysisInputError(
            "'p' must be at least 1."
        )

    return value


def _build_ar_design(
    values: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []

    for index in range(
        p,
        len(values),
    ):
        row = [
            1.0
        ]

        row.extend(
            float(
                values[
                    index - lag
                ]
            )
            for lag in range(
                1,
                p + 1,
            )
        )

        rows.append(
            row
        )

        targets.append(
            float(
                values[index]
            )
        )

    return (
        np.asarray(
            rows,
            dtype=float,
        ),
        np.asarray(
            targets,
            dtype=float,
        ),
    )


def _fit_autoregression(
    values: np.ndarray,
    p: int,
) -> dict[str, Any]:
    design, targets = _build_ar_design(
        values,
        p,
    )

    parameter_count = (
        p + 1
    )

    if (
        len(targets)
        < parameter_count
    ):
        raise InvalidAnalysisInputError(
            "Insufficient observations for AR(p). "
            f"AR({p}) with an intercept requires at least "
            f"{2 * p + 1} total observations."
        )

    coefficients, _, rank, singular_values = (
        np.linalg.lstsq(
            design,
            targets,
            rcond=None,
        )
    )

    intercept = float(
        coefficients[0]
    )

    lag_coefficients = np.asarray(
        coefficients[1:],
        dtype=float,
    )

    fitted = (
        design
        @ coefficients
    )

    residuals = (
        targets
        - fitted
    )

    rss = float(
        np.sum(
            residuals ** 2
        )
    )

    target_mean = float(
        np.mean(
            targets
        )
    )

    tss = float(
        np.sum(
            (
                targets
                - target_mean
            )
            ** 2
        )
    )

    if tss > 0:
        r_squared = float(
            1.0
            - rss / tss
        )

        r_squared_defined = True
    else:
        r_squared = None
        r_squared_defined = False

    residual_degrees_of_freedom = (
        len(targets)
        - int(rank)
    )

    residual_variance = (
        float(
            rss
            / residual_degrees_of_freedom
        )
        if residual_degrees_of_freedom > 0
        else None
    )

    residual_std = (
        float(
            sqrt(
                residual_variance
            )
        )
        if residual_variance is not None
        else None
    )

    mae = float(
        np.mean(
            np.abs(
                residuals
            )
        )
    )

    rmse = float(
        sqrt(
            float(
                np.mean(
                    residuals ** 2
                )
            )
        )
    )

    return {
        "intercept": intercept,
        "lag_coefficients": (
            lag_coefficients
        ),
        "design_rank": int(
            rank
        ),
        "parameter_count": (
            parameter_count
        ),
        "full_rank": (
            int(rank)
            == parameter_count
        ),
        "singular_values": [
            float(value)
            for value in singular_values
        ],
        "training_rows": len(
            targets
        ),
        "targets": targets,
        "fitted": fitted,
        "residuals": residuals,
        "residual_sum_squares": (
            rss
        ),
        "residual_degrees_of_freedom": (
            residual_degrees_of_freedom
        ),
        "residual_variance": (
            residual_variance
        ),
        "residual_std": (
            residual_std
        ),
        "r_squared": (
            r_squared
        ),
        "r_squared_defined": (
            r_squared_defined
        ),
        "mae": mae,
        "rmse": rmse,
    }


def _recursive_ar_forecast(
    values: np.ndarray,
    *,
    intercept: float,
    lag_coefficients: np.ndarray,
    horizon: int,
) -> np.ndarray:
    history = [
        float(value)
        for value in values
    ]

    p = len(
        lag_coefficients
    )

    forecasts: list[float] = []

    for _ in range(
        horizon
    ):
        prediction = (
            intercept
        )

        for lag in range(
            1,
            p + 1,
        ):
            prediction += float(
                lag_coefficients[
                    lag - 1
                ]
                * history[
                    -lag
                ]
            )

        prediction = float(
            prediction
        )

        forecasts.append(
            prediction
        )

        history.append(
            prediction
        )

    return np.asarray(
        forecasts,
        dtype=float,
    )


class AutoregressiveForecast(
    AnalysisMethod
):
    name = "autoregressive_forecast"
    category = "timeseries"
    version = "1.0"

    description = (
        "Fit an AR(p) model with an intercept using "
        "ordinary least squares and produce recursive forecasts."
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
        p = _validate_order(
            kwargs.get(
                "p",
                1,
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

        fit = _fit_autoregression(
            values,
            p,
        )

        forecasts = (
            _recursive_ar_forecast(
                values,
                intercept=fit[
                    "intercept"
                ],
                lag_coefficients=fit[
                    "lag_coefficients"
                ],
                horizon=horizon,
            )
        )

        coefficients = {
            f"lag_{lag}": float(
                fit[
                    "lag_coefficients"
                ][
                    lag - 1
                ]
            )
            for lag in range(
                1,
                p + 1,
            )
        }

        return {
            "model": "autoregressive",
            "order": p,
            "sample_size": len(
                values
            ),
            "training_rows": fit[
                "training_rows"
            ],
            "horizon": horizon,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "intercept": fit[
                "intercept"
            ],
            "coefficients": (
                coefficients
            ),
            "design_rank": fit[
                "design_rank"
            ],
            "parameter_count": fit[
                "parameter_count"
            ],
            "full_rank": fit[
                "full_rank"
            ],
            "r_squared": fit[
                "r_squared"
            ],
            "r_squared_defined": fit[
                "r_squared_defined"
            ],
            "residual_sum_squares": fit[
                "residual_sum_squares"
            ],
            "residual_degrees_of_freedom": fit[
                "residual_degrees_of_freedom"
            ],
            "residual_variance": fit[
                "residual_variance"
            ],
            "residual_std": fit[
                "residual_std"
            ],
            "in_sample_mae": fit[
                "mae"
            ],
            "in_sample_rmse": fit[
                "rmse"
            ],
            "fitted_timestamps": [
                timestamp.isoformat()
                for timestamp in time[
                    p:
                ]
            ],
            "fitted_values": [
                float(value)
                for value in fit[
                    "fitted"
                ]
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
