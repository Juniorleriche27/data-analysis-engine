from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.engines.timeseries.basic import (
    _serialize_time,
    _validate_time_series,
)


def _fit_linear_trend(
    values: np.ndarray,
) -> dict[str, Any]:
    """
    Fit y_t = intercept + slope * t using ordinary
    least squares on the observation index.
    """
    x = np.arange(
        len(values),
        dtype=float,
    )

    x_mean = float(
        np.mean(x)
    )

    y_mean = float(
        np.mean(values)
    )

    centered_x = (
        x - x_mean
    )

    centered_y = (
        values - y_mean
    )

    denominator = float(
        np.sum(
            centered_x ** 2
        )
    )

    slope = float(
        np.sum(
            centered_x
            * centered_y
        )
        / denominator
    )

    intercept = float(
        y_mean
        - slope * x_mean
    )

    fitted = (
        intercept
        + slope * x
    )

    residuals = (
        values
        - fitted
    )

    residual_sum_squares = float(
        np.sum(
            residuals ** 2
        )
    )

    total_sum_squares = float(
        np.sum(
            centered_y ** 2
        )
    )

    if total_sum_squares > 0:
        r_squared: float | None = float(
            1.0
            - (
                residual_sum_squares
                / total_sum_squares
            )
        )

        r_squared_defined = True

    else:
        # R² is not informative when the dependent
        # variable has zero variance.
        r_squared = None
        r_squared_defined = False

    return {
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "residuals": residuals,
        "residual_sum_squares": residual_sum_squares,
        "total_sum_squares": total_sum_squares,
        "r_squared": r_squared,
        "r_squared_defined": r_squared_defined,
    }


class _TrendMethod(AnalysisMethod):
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


class LinearTrend(_TrendMethod):
    name = "linear_trend"
    version = "1.0"

    description = (
        "Estimate a linear time trend using ordinary "
        "least squares on the ordered observations."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        fit = _fit_linear_trend(
            values
        )

        return {
            "sample_size": len(values),
            "time_index_start": 0,
            "time_index_end": (
                len(values) - 1
            ),
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r_squared": fit["r_squared"],
            "r_squared_defined": fit[
                "r_squared_defined"
            ],
            "residual_sum_squares": fit[
                "residual_sum_squares"
            ],
            "timestamps": [
                _serialize_time(value)
                for value in time
            ],
            "fitted_values": [
                float(value)
                for value in fit["fitted"]
            ],
        }


class LinearDetrend(_TrendMethod):
    name = "linear_detrend"
    version = "1.0"

    description = (
        "Remove an estimated ordinary-least-squares "
        "linear trend from an ordered numerical series."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        fit = _fit_linear_trend(
            values
        )

        residuals = fit[
            "residuals"
        ]

        return {
            "sample_size": len(values),
            "slope_removed": fit[
                "slope"
            ],
            "intercept_removed": fit[
                "intercept"
            ],
            "timestamps": [
                _serialize_time(value)
                for value in time
            ],
            "observed": [
                float(value)
                for value in values
            ],
            "fitted_trend": [
                float(value)
                for value in fit["fitted"]
            ],
            "detrended": [
                float(value)
                for value in residuals
            ],
            "detrended_mean": float(
                np.mean(
                    residuals
                )
            ),
            "detrended_std": float(
                np.std(
                    residuals,
                    ddof=1,
                )
            ),
        }
