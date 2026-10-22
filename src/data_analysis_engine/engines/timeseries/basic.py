from math import isfinite
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import (
    AnalysisMethod,
)
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _serialize_time(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


def _validate_time_series(
    time: Any,
    values: Any,
) -> tuple[list[Any], np.ndarray]:
    if not isinstance(
        time,
        (list, tuple, np.ndarray),
    ):
        raise InvalidAnalysisInputError(
            "'time' must be a sequence."
        )

    if not isinstance(
        values,
        (list, tuple, np.ndarray),
    ):
        raise InvalidAnalysisInputError(
            "'values' must be a numerical sequence."
        )

    if len(time) != len(values):
        raise InvalidAnalysisInputError(
            "'time' and 'values' must have the same length."
        )

    if len(values) < 2:
        raise InvalidAnalysisInputError(
            "At least two observations are required."
        )

    numeric_values: list[float] = []

    for value in values:
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
                "'values' must contain only numerical values."
            )

        numeric_value = float(value)

        if not isfinite(numeric_value):
            raise InvalidAnalysisInputError(
                "'values' must contain only finite values."
            )

        numeric_values.append(
            numeric_value
        )

    return (
        list(time),
        np.asarray(
            numeric_values,
            dtype=float,
        ),
    )


class _TimeSeriesMethod(AnalysisMethod):
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

    allow_constant_columns = False


class SimpleReturns(_TimeSeriesMethod):
    name = "simple_returns"
    version = "1.0"

    description = (
        "Compute period-to-period simple returns "
        "for an ordered numerical series."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        previous = values[:-1]

        if np.any(previous == 0):
            raise InvalidAnalysisInputError(
                "Simple returns are undefined when a "
                "previous observation equals zero."
            )

        returns = (
            values[1:] / previous
        ) - 1.0

        return {
            "sample_size": len(values),
            "return_count": len(returns),
            "timestamps": [
                _serialize_time(value)
                for value in time[1:]
            ],
            "returns": [
                float(value)
                for value in returns
            ],
        }


class LogReturns(_TimeSeriesMethod):
    name = "log_returns"
    version = "1.0"

    description = (
        "Compute continuously compounded log returns "
        "for a positive ordered numerical series."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        if np.any(values <= 0):
            raise InvalidAnalysisInputError(
                "Log returns require strictly positive values."
            )

        returns = np.diff(
            np.log(values)
        )

        return {
            "sample_size": len(values),
            "return_count": len(returns),
            "timestamps": [
                _serialize_time(value)
                for value in time[1:]
            ],
            "returns": [
                float(value)
                for value in returns
            ],
        }


class RollingStatistics(_TimeSeriesMethod):
    name = "rolling_statistics"
    version = "1.0"

    description = (
        "Compute rolling mean and sample standard deviation "
        "for an ordered numerical series."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        window = kwargs.get(
            "window",
            3,
        )

        if (
            not isinstance(window, int)
            or isinstance(window, bool)
        ):
            raise InvalidAnalysisInputError(
                "'window' must be an integer."
            )

        if window < 2:
            raise InvalidAnalysisInputError(
                "'window' must be at least 2."
            )

        if window > len(values):
            raise InvalidAnalysisInputError(
                "'window' cannot exceed the series length."
            )

        records: list[dict[str, Any]] = []

        for end in range(
            window,
            len(values) + 1,
        ):
            sample = values[
                end - window:end
            ]

            records.append(
                {
                    "timestamp": _serialize_time(
                        time[end - 1]
                    ),
                    "mean": float(
                        np.mean(sample)
                    ),
                    "std": float(
                        np.std(
                            sample,
                            ddof=1,
                        )
                    ),
                }
            )

        return {
            "sample_size": len(values),
            "window": window,
            "records": records,
        }


class Autocorrelation(_TimeSeriesMethod):
    name = "autocorrelation"
    version = "1.0"

    description = (
        "Measure linear autocorrelation between a series "
        "and a lagged version of itself."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        _, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        lag = kwargs.get(
            "lag",
            1,
        )

        if (
            not isinstance(lag, int)
            or isinstance(lag, bool)
        ):
            raise InvalidAnalysisInputError(
                "'lag' must be an integer."
            )

        if lag < 1:
            raise InvalidAnalysisInputError(
                "'lag' must be at least 1."
            )

        if lag >= len(values):
            raise InvalidAnalysisInputError(
                "'lag' must be smaller than the series length."
            )

        left = values[:-lag]
        right = values[lag:]

        if len(left) < 2:
            raise InvalidAnalysisInputError(
                "At least two paired observations are required "
                "after applying the lag."
            )

        if (
            np.all(left == left[0])
            or np.all(right == right[0])
        ):
            raise InvalidAnalysisInputError(
                "Autocorrelation is undefined for a "
                "constant lagged series."
            )

        coefficient = float(
            np.corrcoef(
                left,
                right,
            )[0, 1]
        )

        return {
            "lag": lag,
            "paired_observations": len(left),
            "coefficient": coefficient,
        }


class LaggedSeries(_TimeSeriesMethod):
    name = "lagged_series"
    version = "1.0"

    description = (
        "Construct aligned current and lagged observations "
        "for an ordered numerical time series."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        lag = kwargs.get(
            "lag",
            1,
        )

        if (
            not isinstance(lag, int)
            or isinstance(lag, bool)
        ):
            raise InvalidAnalysisInputError(
                "'lag' must be an integer."
            )

        if lag < 1:
            raise InvalidAnalysisInputError(
                "'lag' must be at least 1."
            )

        if lag >= len(values):
            raise InvalidAnalysisInputError(
                "'lag' must be smaller than the series length."
            )

        records: list[dict[str, Any]] = []

        for index in range(
            lag,
            len(values),
        ):
            records.append(
                {
                    "timestamp": _serialize_time(
                        time[index]
                    ),
                    "value": float(
                        values[index]
                    ),
                    "lagged_timestamp": _serialize_time(
                        time[index - lag]
                    ),
                    "lagged_value": float(
                        values[index - lag]
                    ),
                }
            )

        return {
            "lag": lag,
            "sample_size": len(values),
            "paired_observations": len(records),
            "records": records,
        }


class AutocorrelationFunction(_TimeSeriesMethod):
    name = "autocorrelation_function"
    version = "1.0"

    description = (
        "Compute sample autocorrelation coefficients "
        "for multiple positive lags."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        _, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        default_max_lag = min(
            10,
            len(values) - 2,
        )

        max_lag = kwargs.get(
            "max_lag",
            default_max_lag,
        )

        if (
            not isinstance(max_lag, int)
            or isinstance(max_lag, bool)
        ):
            raise InvalidAnalysisInputError(
                "'max_lag' must be an integer."
            )

        if max_lag < 1:
            raise InvalidAnalysisInputError(
                "'max_lag' must be at least 1."
            )

        if max_lag > len(values) - 2:
            raise InvalidAnalysisInputError(
                "'max_lag' must leave at least two "
                "paired observations."
            )

        coefficients: list[dict[str, Any]] = []

        for lag in range(
            1,
            max_lag + 1,
        ):
            left = values[:-lag]
            right = values[lag:]

            defined = not (
                np.all(left == left[0])
                or np.all(right == right[0])
            )

            coefficient = (
                float(
                    np.corrcoef(
                        left,
                        right,
                    )[0, 1]
                )
                if defined
                else None
            )

            coefficients.append(
                {
                    "lag": lag,
                    "paired_observations": len(left),
                    "coefficient": coefficient,
                    "defined": defined,
                }
            )

        return {
            "sample_size": len(values),
            "max_lag": max_lag,
            "autocorrelations": coefficients,
        }
