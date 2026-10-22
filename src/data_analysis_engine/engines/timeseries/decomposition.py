from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.basic import (
    _serialize_time,
    _validate_time_series,
)


def _validate_window(
    window: Any,
    series_length: int,
    *,
    parameter_name: str = "window",
) -> int:
    if (
        not isinstance(window, int)
        or isinstance(window, bool)
    ):
        raise InvalidAnalysisInputError(
            f"'{parameter_name}' must be an integer."
        )

    if window < 2:
        raise InvalidAnalysisInputError(
            f"'{parameter_name}' must be at least 2."
        )

    if window >= series_length:
        raise InvalidAnalysisInputError(
            f"'{parameter_name}' must be smaller than "
            "the series length."
        )

    return window


def _centered_moving_average(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Compute a centered moving average aligned to the
    original observations.

    Odd windows use a direct centered average.

    Even windows use a two-stage moving average so the
    resulting trend is centered on an actual observation.
    """
    n = len(values)

    trend = np.full(
        n,
        np.nan,
        dtype=float,
    )

    if window % 2 == 1:
        half = window // 2

        for center in range(
            half,
            n - half,
        ):
            trend[center] = float(
                np.mean(
                    values[
                        center - half:
                        center + half + 1
                    ]
                )
            )

        return trend

    moving_averages = np.asarray(
        [
            np.mean(
                values[
                    start:
                    start + window
                ]
            )
            for start in range(
                n - window + 1
            )
        ],
        dtype=float,
    )

    centered = (
        moving_averages[:-1]
        + moving_averages[1:]
    ) / 2.0

    start_index = window // 2

    for offset, value in enumerate(
        centered
    ):
        trend[
            start_index + offset
        ] = float(value)

    return trend


def _optional_float(
    value: float,
) -> float | None:
    if np.isfinite(value):
        return float(value)

    return None


class _DecompositionMethod(AnalysisMethod):
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


class MovingAverageTrend(_DecompositionMethod):
    name = "moving_average_trend"
    version = "1.0"

    description = (
        "Estimate the trend component of an ordered "
        "numerical time series using a centered moving average."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        window = _validate_window(
            kwargs.get(
                "window",
                3,
            ),
            len(values),
        )

        trend = _centered_moving_average(
            values,
            window,
        )

        return {
            "sample_size": len(values),
            "window": window,
            "centered": True,
            "timestamps": [
                _serialize_time(value)
                for value in time
            ],
            "trend": [
                _optional_float(value)
                for value in trend
            ],
        }


class AdditiveSeasonalDecomposition(
    _DecompositionMethod
):
    name = "additive_seasonal_decomposition"
    version = "1.0"

    description = (
        "Decompose an ordered numerical series into "
        "trend, seasonal, and residual components using "
        "an additive moving-average decomposition."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        period = _validate_window(
            kwargs.get("period"),
            len(values),
            parameter_name="period",
        )

        if len(values) < 2 * period:
            raise InvalidAnalysisInputError(
                "Additive seasonal decomposition requires "
                "at least two complete seasonal periods."
            )

        trend = _centered_moving_average(
            values,
            period,
        )

        detrended = np.full(
            len(values),
            np.nan,
            dtype=float,
        )

        valid_trend = np.isfinite(
            trend
        )

        detrended[
            valid_trend
        ] = (
            values[valid_trend]
            - trend[valid_trend]
        )

        seasonal_pattern = np.zeros(
            period,
            dtype=float,
        )

        for phase in range(
            period
        ):
            phase_values = np.asarray(
                [
                    detrended[index]
                    for index in range(
                        phase,
                        len(values),
                        period,
                    )
                    if np.isfinite(
                        detrended[index]
                    )
                ],
                dtype=float,
            )

            if len(phase_values) == 0:
                raise InvalidAnalysisInputError(
                    "Insufficient observations to estimate "
                    "every seasonal phase."
                )

            seasonal_pattern[
                phase
            ] = float(
                np.mean(
                    phase_values
                )
            )

        # Additive seasonal components must be centered
        # around zero.
        seasonal_pattern = (
            seasonal_pattern
            - np.mean(
                seasonal_pattern
            )
        )

        seasonal = np.asarray(
            [
                seasonal_pattern[
                    index % period
                ]
                for index in range(
                    len(values)
                )
            ],
            dtype=float,
        )

        residual = np.full(
            len(values),
            np.nan,
            dtype=float,
        )

        residual[
            valid_trend
        ] = (
            values[valid_trend]
            - trend[valid_trend]
            - seasonal[valid_trend]
        )

        return {
            "model": "additive",
            "sample_size": len(values),
            "period": period,
            "timestamps": [
                _serialize_time(value)
                for value in time
            ],
            "observed": [
                float(value)
                for value in values
            ],
            "trend": [
                _optional_float(value)
                for value in trend
            ],
            "seasonal": [
                float(value)
                for value in seasonal
            ],
            "seasonal_pattern": [
                float(value)
                for value in seasonal_pattern
            ],
            "residual": [
                _optional_float(value)
                for value in residual
            ],
        }
