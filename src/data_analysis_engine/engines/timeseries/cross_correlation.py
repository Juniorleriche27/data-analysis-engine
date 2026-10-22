from math import isfinite
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_paired_series(
    x: Any,
    y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(
        x,
        (list, tuple, np.ndarray),
    ):
        raise InvalidAnalysisInputError(
            "'x' must be a numerical sequence."
        )

    if not isinstance(
        y,
        (list, tuple, np.ndarray),
    ):
        raise InvalidAnalysisInputError(
            "'y' must be a numerical sequence."
        )

    if len(x) != len(y):
        raise InvalidAnalysisInputError(
            "'x' and 'y' must have the same length."
        )

    if len(x) < 3:
        raise InvalidAnalysisInputError(
            "At least 3 paired observations are required."
        )

    def validate(
        values: list | tuple | np.ndarray,
        name: str,
    ) -> np.ndarray:
        validated: list[float] = []

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
                    f"'{name}' must contain only numerical values."
                )

            numeric_value = float(value)

            if not isfinite(numeric_value):
                raise InvalidAnalysisInputError(
                    f"'{name}' must contain only finite values."
                )

            validated.append(
                numeric_value
            )

        return np.asarray(
            validated,
            dtype=float,
        )

    return (
        validate(x, "x"),
        validate(y, "y"),
    )


class CrossCorrelation(AnalysisMethod):
    name = "cross_correlation"
    category = "timeseries"
    version = "1.0"

    description = (
        "Compute cross-correlation between two ordered "
        "numerical series across positive and negative lags."
    )

    supported_roles = (
        "time",
        "measure",
    )

    minimum_compatible_columns = 3

    selection_mode = "time_series_pairwise"
    analysis_unit = "series_pair"

    role_requirements = (
        ("time", 1),
        ("measure", 2),
    )

    allow_constant_columns = False

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_paired_series(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        max_lag = kwargs.get(
            "max_lag",
            min(
                10,
                len(x) - 2,
            ),
        )

        if (
            not isinstance(max_lag, int)
            or isinstance(max_lag, bool)
        ):
            raise InvalidAnalysisInputError(
                "'max_lag' must be an integer."
            )

        if max_lag < 0:
            raise InvalidAnalysisInputError(
                "'max_lag' must be greater than or equal to 0."
            )

        if max_lag > len(x) - 2:
            raise InvalidAnalysisInputError(
                "'max_lag' must leave at least two "
                "paired observations."
            )

        correlations: list[
            dict[str, Any]
        ] = []

        for lag in range(
            -max_lag,
            max_lag + 1,
        ):
            if lag < 0:
                shift = abs(lag)

                x_aligned = x[
                    shift:
                ]

                y_aligned = y[
                    :-shift
                ]

            elif lag > 0:
                x_aligned = x[
                    :-lag
                ]

                y_aligned = y[
                    lag:
                ]

            else:
                x_aligned = x
                y_aligned = y

            defined = not (
                np.all(
                    x_aligned
                    == x_aligned[0]
                )
                or np.all(
                    y_aligned
                    == y_aligned[0]
                )
            )

            coefficient = (
                float(
                    np.corrcoef(
                        x_aligned,
                        y_aligned,
                    )[0, 1]
                )
                if defined
                else None
            )

            correlations.append(
                {
                    "lag": lag,
                    "paired_observations": len(
                        x_aligned
                    ),
                    "coefficient": coefficient,
                    "defined": defined,
                }
            )

        defined_results = [
            item
            for item in correlations
            if item["defined"]
        ]

        strongest = None

        if defined_results:
            strongest_item = sorted(
                defined_results,
                key=lambda item: (
                    -abs(
                        item["coefficient"]
                    ),
                    abs(
                        item["lag"]
                    ),
                    item["lag"],
                ),
            )[0]

            lag = strongest_item[
                "lag"
            ]

            if lag > 0:
                direction = "x_leads_y"
            elif lag < 0:
                direction = "y_leads_x"
            else:
                direction = "simultaneous"

            strongest = {
                "lag": lag,
                "coefficient": strongest_item[
                    "coefficient"
                ],
                "absolute_coefficient": abs(
                    strongest_item[
                        "coefficient"
                    ]
                ),
                "direction": direction,
            }

        return {
            "sample_size": len(x),
            "max_lag": max_lag,
            "lag_convention": {
                "positive": "x_leads_y",
                "zero": "simultaneous",
                "negative": "y_leads_x",
            },
            "correlations": correlations,
            "strongest_relationship": strongest,
        }
