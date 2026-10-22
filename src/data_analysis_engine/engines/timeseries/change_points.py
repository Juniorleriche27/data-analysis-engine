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


def _validate_positive_threshold(
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


class MeanShiftChangePoint(AnalysisMethod):
    name = "mean_shift_change_point"
    category = "timeseries"
    version = "1.0"

    description = (
        "Detect the strongest single mean-level change point "
        "in an ordered numerical time series."
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
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        min_segment_size = kwargs.get(
            "min_segment_size",
            2,
        )

        if (
            not isinstance(min_segment_size, int)
            or isinstance(min_segment_size, bool)
        ):
            raise InvalidAnalysisInputError(
                "'min_segment_size' must be an integer."
            )

        if min_segment_size < 2:
            raise InvalidAnalysisInputError(
                "'min_segment_size' must be at least 2."
            )

        if (
            len(values)
            < 2 * min_segment_size
        ):
            raise InvalidAnalysisInputError(
                "The series must contain at least two "
                "segments of 'min_segment_size' observations."
            )

        threshold = _validate_positive_threshold(
            kwargs.get(
                "threshold",
                1.0,
            ),
            "threshold",
        )

        overall_std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        overall_mean = float(
            np.mean(values)
        )

        total_sse = float(
            np.sum(
                (
                    values
                    - overall_mean
                )
                ** 2
            )
        )

        candidates: list[
            dict[str, Any]
        ] = []

        for split_index in range(
            min_segment_size,
            len(values)
            - min_segment_size
            + 1,
        ):
            left = values[
                :split_index
            ]

            right = values[
                split_index:
            ]

            left_mean = float(
                np.mean(left)
            )

            right_mean = float(
                np.mean(right)
            )

            absolute_mean_shift = abs(
                right_mean
                - left_mean
            )

            if overall_std > 0:
                standardized_mean_shift = float(
                    absolute_mean_shift
                    / overall_std
                )
            else:
                standardized_mean_shift = 0.0

            left_sse = float(
                np.sum(
                    (
                        left
                        - left_mean
                    )
                    ** 2
                )
            )

            right_sse = float(
                np.sum(
                    (
                        right
                        - right_mean
                    )
                    ** 2
                )
            )

            segmented_sse = (
                left_sse
                + right_sse
            )

            sse_reduction = (
                total_sse
                - segmented_sse
            )

            if total_sse > 0:
                relative_sse_reduction = float(
                    sse_reduction
                    / total_sse
                )
            else:
                relative_sse_reduction = 0.0

            candidates.append(
                {
                    "split_index": split_index,
                    "timestamp": _serialize_time(
                        time[
                            split_index
                        ]
                    ),
                    "left_size": len(left),
                    "right_size": len(right),
                    "left_mean": left_mean,
                    "right_mean": right_mean,
                    "absolute_mean_shift": (
                        absolute_mean_shift
                    ),
                    "standardized_mean_shift": (
                        standardized_mean_shift
                    ),
                    "relative_sse_reduction": (
                        relative_sse_reduction
                    ),
                }
            )

        strongest = max(
            candidates,
            key=lambda item: (
                item[
                    "standardized_mean_shift"
                ],
                item[
                    "relative_sse_reduction"
                ],
            ),
        )

        detected = (
            strongest[
                "standardized_mean_shift"
            ]
            >= threshold
        )

        return {
            "diagnostic_type": (
                "single_mean_shift_scan"
            ),
            "sample_size": len(values),
            "min_segment_size": (
                min_segment_size
            ),
            "threshold": threshold,
            "candidates_evaluated": len(
                candidates
            ),
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "detected": detected,
            "change_index": (
                strongest[
                    "split_index"
                ]
                if detected
                else None
            ),
            "change_timestamp": (
                strongest[
                    "timestamp"
                ]
                if detected
                else None
            ),
            "strongest_candidate": (
                strongest
            ),
            "limitations": [
                (
                    "This method detects one strongest "
                    "mean-level change only."
                ),
                (
                    "It does not by itself detect variance, "
                    "volatility, trend, or multiple regime changes."
                ),
            ],
        }
