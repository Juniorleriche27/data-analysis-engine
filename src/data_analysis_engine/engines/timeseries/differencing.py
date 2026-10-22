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


class _DifferencingMethod(AnalysisMethod):
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


class FirstDifference(_DifferencingMethod):
    name = "first_difference"
    version = "1.0"

    description = (
        "Compute first differences of an ordered "
        "numerical time series."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        differences = np.diff(
            values
        )

        return {
            "order": 1,
            "sample_size": len(values),
            "difference_count": len(
                differences
            ),
            "timestamps": [
                _serialize_time(value)
                for value in time[1:]
            ],
            "differences": [
                float(value)
                for value in differences
            ],
        }


class SeasonalDifference(_DifferencingMethod):
    name = "seasonal_difference"
    version = "1.0"

    description = (
        "Compute lag-k seasonal differences of an "
        "ordered numerical time series."
    )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        period = kwargs.get(
            "period"
        )

        if (
            not isinstance(period, int)
            or isinstance(period, bool)
        ):
            raise InvalidAnalysisInputError(
                "'period' must be an integer."
            )

        if period < 1:
            raise InvalidAnalysisInputError(
                "'period' must be at least 1."
            )

        if period >= len(values):
            raise InvalidAnalysisInputError(
                "'period' must be smaller than "
                "the series length."
            )

        differences = (
            values[period:]
            - values[:-period]
        )

        return {
            "period": period,
            "sample_size": len(values),
            "difference_count": len(
                differences
            ),
            "timestamps": [
                _serialize_time(value)
                for value in time[
                    period:
                ]
            ],
            "differences": [
                float(value)
                for value in differences
            ],
        }
