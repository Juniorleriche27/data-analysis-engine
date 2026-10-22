from math import isfinite
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_sequence(
    values: Any,
    name: str,
) -> np.ndarray:
    if not isinstance(
        values,
        (list, tuple, np.ndarray),
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be a numerical sequence."
        )

    if len(values) == 0:
        raise InvalidAnalysisInputError(
            f"'{name}' cannot be empty."
        )

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

        validated.append(numeric_value)

    return np.asarray(
        validated,
        dtype=float,
    )


class DynamicTimeWarping(AnalysisMethod):
    name = "dynamic_time_warping"
    category = "timeseries"
    version = "1.0"

    description = (
        "Measure similarity between two ordered numerical "
        "time series using Dynamic Time Warping."
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

    allow_constant_columns = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x = _validate_sequence(
            kwargs.get("x"),
            "x",
        )

        y = _validate_sequence(
            kwargs.get("y"),
            "y",
        )

        n = len(x)
        m = len(y)

        window = kwargs.get(
            "window"
        )

        if window is None:
            effective_window = max(
                n,
                m,
            )

        else:
            if (
                not isinstance(window, int)
                or isinstance(window, bool)
            ):
                raise InvalidAnalysisInputError(
                    "'window' must be an integer or None."
                )

            if window < 0:
                raise InvalidAnalysisInputError(
                    "'window' must be greater than or equal to 0."
                )

            if window < abs(n - m):
                raise InvalidAnalysisInputError(
                    "'window' is too small for sequences "
                    "with different lengths."
                )

            effective_window = window

        accumulated = np.full(
            (
                n + 1,
                m + 1,
            ),
            np.inf,
            dtype=float,
        )

        accumulated[0, 0] = 0.0

        for i in range(
            1,
            n + 1,
        ):
            start = max(
                1,
                i - effective_window,
            )

            end = min(
                m,
                i + effective_window,
            )

            for j in range(
                start,
                end + 1,
            ):
                local_cost = abs(
                    x[i - 1]
                    - y[j - 1]
                )

                accumulated[i, j] = (
                    local_cost
                    + min(
                        accumulated[i - 1, j],
                        accumulated[i, j - 1],
                        accumulated[i - 1, j - 1],
                    )
                )

        distance = float(
            accumulated[n, m]
        )

        if not isfinite(distance):
            raise InvalidAnalysisInputError(
                "No valid DTW alignment could be found."
            )

        path: list[list[int]] = []

        i = n
        j = m

        while i > 0 and j > 0:
            path.append(
                [
                    i - 1,
                    j - 1,
                ]
            )

            diagonal = accumulated[
                i - 1,
                j - 1,
            ]

            up = accumulated[
                i - 1,
                j,
            ]

            left = accumulated[
                i,
                j - 1,
            ]

            move = int(
                np.argmin(
                    [
                        diagonal,
                        up,
                        left,
                    ]
                )
            )

            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1

        path.reverse()

        path_length = len(path)

        return {
            "distance": distance,
            "normalized_distance": (
                distance / path_length
                if path_length
                else 0.0
            ),
            "x_length": n,
            "y_length": m,
            "window": (
                None
                if window is None
                else window
            ),
            "path_length": path_length,
            "warping_path": path,
        }
