from math import sqrt
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


def _validate_positive_number(
    value: Any,
    name: str,
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
            f"'{name}' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be finite and greater than zero."
        )

    return value


def _prepare_volatility_series(
    time: Any,
    values: Any,
    *,
    input_kind: str,
    return_type: str,
) -> tuple[list[Any], np.ndarray, str]:
    time_values, numeric_values = _validate_time_series(
        time,
        values,
    )

    if input_kind not in {
        "returns",
        "prices",
    }:
        raise InvalidAnalysisInputError(
            "'input_kind' must be either "
            "'returns' or 'prices'."
        )

    if return_type not in {
        "simple",
        "log",
    }:
        raise InvalidAnalysisInputError(
            "'return_type' must be either "
            "'simple' or 'log'."
        )

    if input_kind == "returns":
        return (
            time_values,
            numeric_values,
            "none",
        )

    if return_type == "simple":
        previous = numeric_values[:-1]

        if np.any(previous == 0):
            raise InvalidAnalysisInputError(
                "Simple returns are undefined when a "
                "previous price equals zero."
            )

        transformed = (
            numeric_values[1:]
            / previous
        ) - 1.0

        transformation = "simple_returns"

    else:
        if np.any(
            numeric_values <= 0
        ):
            raise InvalidAnalysisInputError(
                "Log returns require strictly positive prices."
            )

        transformed = np.diff(
            np.log(
                numeric_values
            )
        )

        transformation = "log_returns"

    return (
        time_values[1:],
        transformed,
        transformation,
    )


class _VolatilityMethod(AnalysisMethod):
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


class RollingVolatility(_VolatilityMethod):
    name = "rolling_volatility"
    version = "1.0"

    description = (
        "Compute rolling sample volatility from returns "
        "or from prices converted to simple or log returns."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        input_kind = kwargs.get(
            "input_kind",
            "returns",
        )

        return_type = kwargs.get(
            "return_type",
            "log",
        )

        time, values, transformation = (
            _prepare_volatility_series(
                kwargs.get("time"),
                kwargs.get("values"),
                input_kind=input_kind,
                return_type=return_type,
            )
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
                "'window' cannot exceed the number "
                "of available return observations."
            )

        annualization_factor = kwargs.get(
            "annualization_factor"
        )

        multiplier = 1.0

        if annualization_factor is not None:
            annualization_factor = (
                _validate_positive_number(
                    annualization_factor,
                    "annualization_factor",
                )
            )

            multiplier = sqrt(
                annualization_factor
            )

        records: list[
            dict[str, Any]
        ] = []

        for end in range(
            window,
            len(values) + 1,
        ):
            sample = values[
                end - window:
                end
            ]

            raw_volatility = float(
                np.std(
                    sample,
                    ddof=1,
                )
            )

            records.append(
                {
                    "timestamp": _serialize_time(
                        time[
                            end - 1
                        ]
                    ),
                    "volatility": (
                        raw_volatility
                        * multiplier
                    ),
                    "raw_volatility": (
                        raw_volatility
                    ),
                }
            )

        return {
            "input_kind": input_kind,
            "return_type": return_type,
            "transformation": transformation,
            "sample_size": len(values),
            "window": window,
            "annualization_factor": (
                annualization_factor
            ),
            "records": records,
        }


class EWMAVolatility(_VolatilityMethod):
    name = "ewma_volatility"
    version = "1.0"

    description = (
        "Estimate exponentially weighted volatility "
        "from returns or prices using recursive EWMA variance."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        input_kind = kwargs.get(
            "input_kind",
            "returns",
        )

        return_type = kwargs.get(
            "return_type",
            "log",
        )

        time, values, transformation = (
            _prepare_volatility_series(
                kwargs.get("time"),
                kwargs.get("values"),
                input_kind=input_kind,
                return_type=return_type,
            )
        )

        decay = kwargs.get(
            "decay",
            0.94,
        )

        if (
            not isinstance(
                decay,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
            or isinstance(
                decay,
                (bool, np.bool_),
            )
        ):
            raise InvalidAnalysisInputError(
                "'decay' must be numerical."
            )

        decay = float(
            decay
        )

        if (
            not np.isfinite(decay)
            or decay <= 0
            or decay >= 1
        ):
            raise InvalidAnalysisInputError(
                "'decay' must be strictly between 0 and 1."
            )

        annualization_factor = kwargs.get(
            "annualization_factor"
        )

        multiplier = 1.0

        if annualization_factor is not None:
            annualization_factor = (
                _validate_positive_number(
                    annualization_factor,
                    "annualization_factor",
                )
            )

            multiplier = sqrt(
                annualization_factor
            )

        variance = float(
            values[0] ** 2
        )

        records: list[
            dict[str, Any]
        ] = [
            {
                "timestamp": _serialize_time(
                    time[0]
                ),
                "variance": variance,
                "volatility": (
                    sqrt(variance)
                    * multiplier
                ),
            }
        ]

        for index in range(
            1,
            len(values),
        ):
            variance = float(
                decay
                * variance
                + (
                    1.0 - decay
                )
                * (
                    values[index]
                    ** 2
                )
            )

            records.append(
                {
                    "timestamp": _serialize_time(
                        time[index]
                    ),
                    "variance": variance,
                    "volatility": (
                        sqrt(variance)
                        * multiplier
                    ),
                }
            )

        return {
            "input_kind": input_kind,
            "return_type": return_type,
            "transformation": transformation,
            "sample_size": len(values),
            "decay": decay,
            "annualization_factor": (
                annualization_factor
            ),
            "initial_variance": float(
                values[0] ** 2
            ),
            "records": records,
            "assumption": (
                "EWMA variance is computed around zero mean."
            ),
        }
