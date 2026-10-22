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


def _validate_variance_ratio_threshold(
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
            "'variance_ratio_threshold' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'variance_ratio_threshold' must be finite "
            "and greater than 1."
        )

    return value


class VarianceShiftChangePoint(AnalysisMethod):
    name = "variance_shift_change_point"
    category = "timeseries"
    version = "1.0"

    description = (
        "Detect the strongest single variance-level change "
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
            3,
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

        threshold = (
            _validate_variance_ratio_threshold(
                kwargs.get(
                    "variance_ratio_threshold",
                    4.0,
                )
            )
        )

        overall_variance = float(
            np.var(
                values,
                ddof=1,
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

            left_variance = float(
                np.var(
                    left,
                    ddof=1,
                )
            )

            right_variance = float(
                np.var(
                    right,
                    ddof=1,
                )
            )

            minimum_variance = min(
                left_variance,
                right_variance,
            )

            maximum_variance = max(
                left_variance,
                right_variance,
            )

            if minimum_variance > 0:
                variance_ratio = float(
                    maximum_variance
                    / minimum_variance
                )

                variance_ratio_defined = True

            elif maximum_variance == 0:
                variance_ratio = 1.0
                variance_ratio_defined = True

            else:
                # One segment has exactly zero variance
                # while the other does not. The theoretical
                # variance ratio is infinite.
                variance_ratio = None
                variance_ratio_defined = False

            absolute_variance_shift = abs(
                right_variance
                - left_variance
            )

            if overall_variance > 0:
                normalized_variance_shift = float(
                    absolute_variance_shift
                    / overall_variance
                )
            else:
                normalized_variance_shift = 0.0

            balance = float(
                2.0
                * min(
                    len(left),
                    len(right),
                )
                / len(values)
            )

            selection_score = float(
                normalized_variance_shift
                * balance
            )

            if right_variance > left_variance:
                direction = "variance_increase"
            elif right_variance < left_variance:
                direction = "variance_decrease"
            else:
                direction = "no_variance_change"

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
                    "left_variance": left_variance,
                    "right_variance": right_variance,
                    "absolute_variance_shift": (
                        absolute_variance_shift
                    ),
                    "normalized_variance_shift": (
                        normalized_variance_shift
                    ),
                    "variance_ratio": variance_ratio,
                    "variance_ratio_defined": (
                        variance_ratio_defined
                    ),
                    "balance": balance,
                    "selection_score": (
                        selection_score
                    ),
                    "direction": direction,
                }
            )

        strongest = max(
            candidates,
            key=lambda item: (
                item[
                    "selection_score"
                ],
                item[
                    "normalized_variance_shift"
                ],
            ),
        )

        if strongest[
            "variance_ratio_defined"
        ]:
            detected = (
                strongest[
                    "variance_ratio"
                ]
                >= threshold
            )

        else:
            detected = (
                strongest[
                    "left_variance"
                ]
                != strongest[
                    "right_variance"
                ]
            )

        return {
            "diagnostic_type": (
                "single_variance_shift_scan"
            ),
            "sample_size": len(values),
            "min_segment_size": (
                min_segment_size
            ),
            "variance_ratio_threshold": (
                threshold
            ),
            "overall_variance": (
                overall_variance
            ),
            "candidates_evaluated": len(
                candidates
            ),
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
            "direction": (
                strongest[
                    "direction"
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
                    "variance-level change only."
                ),
                (
                    "It is a descriptive regime diagnostic "
                    "and does not provide a formal structural "
                    "break p-value."
                ),
            ],
        }
