from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _serialize_timestamp(
    value: pd.Timestamp,
) -> str:
    return value.isoformat()


def _validate_ratio(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float | None = None,
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

    if not np.isfinite(value):
        raise InvalidAnalysisInputError(
            f"'{name}' must be finite."
        )

    if value < minimum:
        raise InvalidAnalysisInputError(
            f"'{name}' must be at least {minimum}."
        )

    if (
        maximum is not None
        and value > maximum
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must not exceed {maximum}."
        )

    return value


class TemporalStructureDiagnostic(
    AnalysisMethod
):
    name = "temporal_structure_diagnostic"
    category = "timeseries"
    version = "1.0"

    description = (
        "Diagnose ordering, duplicate timestamps, "
        "interval regularity, and temporal gaps."
    )

    supported_roles = (
        "time",
    )

    minimum_compatible_columns = 1

    selection_mode = "time_axis"
    analysis_unit = "timeline"

    role_requirements = (
        ("time", 1),
    )

    allow_constant_columns = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        time = kwargs.get(
            "time"
        )

        if not isinstance(
            time,
            (
                list,
                tuple,
                np.ndarray,
                pd.Series,
                pd.Index,
            ),
        ):
            raise InvalidAnalysisInputError(
                "'time' must be a sequence."
            )

        if len(time) < 2:
            raise InvalidAnalysisInputError(
                "At least two timestamps are required."
            )

        parsed = pd.to_datetime(
            list(time),
            errors="coerce",
            utc=True,
        )

        if parsed.isna().any():
            raise InvalidAnalysisInputError(
                "All values in 'time' must be valid timestamps."
            )

        tolerance_ratio = _validate_ratio(
            kwargs.get(
                "tolerance_ratio",
                0.05,
            ),
            "tolerance_ratio",
            minimum=0.0,
            maximum=1.0,
        )

        gap_factor = _validate_ratio(
            kwargs.get(
                "gap_factor",
                1.5,
            ),
            "gap_factor",
            minimum=1.0,
        )

        # Timestamp.value is explicitly expressed in nanoseconds.
        # Do not rely on DatetimeIndex.asi8 because its unit can
        # follow the underlying datetime resolution in newer pandas.
        original_ns = np.asarray(
            [
                timestamp.value
                for timestamp in parsed
            ],
            dtype=np.int64,
        )

        input_monotonic_increasing = bool(
            parsed.is_monotonic_increasing
        )

        input_strictly_increasing = bool(
            np.all(
                np.diff(
                    original_ns
                )
                > 0
            )
        )

        order = np.argsort(
            original_ns,
            kind="stable",
        )

        sorted_ns = original_ns[
            order
        ]

        sorted_time = parsed[
            order
        ]

        deltas_ns = np.diff(
            sorted_ns
        )

        duplicate_observation_count = int(
            len(sorted_ns)
            - len(
                np.unique(
                    sorted_ns
                )
            )
        )

        positive_deltas = deltas_ns[
            deltas_ns > 0
        ]

        if len(
            positive_deltas
        ) == 0:
            return {
                "diagnostic_type": (
                    "temporal_structure"
                ),
                "sample_size": len(parsed),
                "input_monotonic_increasing": (
                    input_monotonic_increasing
                ),
                "input_strictly_increasing": (
                    input_strictly_increasing
                ),
                "duplicate_observation_count": (
                    duplicate_observation_count
                ),
                "expected_interval_seconds": None,
                "regularity_ratio": 0.0,
                "irregular_interval_count": (
                    len(deltas_ns)
                ),
                "gap_count": 0,
                "estimated_missing_periods": 0,
                "status": "degenerate_time_axis",
                "intervals": [],
                "warnings": [
                    (
                        "No positive temporal interval "
                        "could be inferred."
                    )
                ],
            }

        expected_interval_ns = float(
            np.median(
                positive_deltas
            )
        )

        expected_interval_seconds = (
            expected_interval_ns
            / 1_000_000_000.0
        )

        intervals: list[
            dict[str, Any]
        ] = []

        regular_count = 0
        irregular_count = 0
        gap_count = 0
        estimated_missing_periods = 0

        for index, delta_ns in enumerate(
            deltas_ns
        ):
            delta_ns_float = float(
                delta_ns
            )

            interval_seconds = (
                delta_ns_float
                / 1_000_000_000.0
            )

            relative_deviation = float(
                abs(
                    delta_ns_float
                    - expected_interval_ns
                )
                / expected_interval_ns
            )

            duplicate = (
                delta_ns == 0
            )

            gap = bool(
                delta_ns_float
                > (
                    expected_interval_ns
                    * gap_factor
                )
            )

            regular = bool(
                not duplicate
                and relative_deviation
                <= tolerance_ratio
            )

            if regular:
                regular_count += 1
            else:
                irregular_count += 1

            missing_periods = 0

            if gap:
                gap_count += 1

                missing_periods = max(
                    int(
                        round(
                            delta_ns_float
                            / expected_interval_ns
                        )
                    )
                    - 1,
                    0,
                )

                estimated_missing_periods += (
                    missing_periods
                )

            if duplicate:
                interval_type = "duplicate"
            elif gap:
                interval_type = "gap"
            elif regular:
                interval_type = "regular"
            else:
                interval_type = "irregular"

            intervals.append(
                {
                    "from_timestamp": (
                        _serialize_timestamp(
                            sorted_time[
                                index
                            ]
                        )
                    ),
                    "to_timestamp": (
                        _serialize_timestamp(
                            sorted_time[
                                index + 1
                            ]
                        )
                    ),
                    "interval_seconds": (
                        interval_seconds
                    ),
                    "relative_deviation": (
                        relative_deviation
                    ),
                    "type": interval_type,
                    "estimated_missing_periods": (
                        missing_periods
                    ),
                }
            )

        interval_count = len(
            deltas_ns
        )

        regularity_ratio = float(
            regular_count
            / interval_count
        )

        if (
            duplicate_observation_count
            == 0
            and irregular_count == 0
        ):
            status = "regular"

        elif regularity_ratio >= 0.8:
            status = "mostly_regular"

        else:
            status = "irregular"

        time_span_seconds = float(
            (
                sorted_ns[-1]
                - sorted_ns[0]
            )
            / 1_000_000_000.0
        )

        warnings: list[str] = []

        if not input_monotonic_increasing:
            warnings.append(
                "Input timestamps were not in chronological order."
            )

        if duplicate_observation_count:
            warnings.append(
                "Duplicate timestamps were detected."
            )

        if gap_count:
            warnings.append(
                "Temporal gaps were detected."
            )

        if irregular_count:
            warnings.append(
                "Irregular temporal intervals were detected."
            )

        return {
            "diagnostic_type": (
                "temporal_structure"
            ),
            "sample_size": len(parsed),
            "input_monotonic_increasing": (
                input_monotonic_increasing
            ),
            "input_strictly_increasing": (
                input_strictly_increasing
            ),
            "first_timestamp": (
                _serialize_timestamp(
                    sorted_time[0]
                )
            ),
            "last_timestamp": (
                _serialize_timestamp(
                    sorted_time[-1]
                )
            ),
            "time_span_seconds": (
                time_span_seconds
            ),
            "duplicate_observation_count": (
                duplicate_observation_count
            ),
            "expected_interval_seconds": (
                expected_interval_seconds
            ),
            "tolerance_ratio": (
                tolerance_ratio
            ),
            "gap_factor": (
                gap_factor
            ),
            "regular_interval_count": (
                regular_count
            ),
            "irregular_interval_count": (
                irregular_count
            ),
            "regularity_ratio": (
                regularity_ratio
            ),
            "gap_count": gap_count,
            "estimated_missing_periods": (
                estimated_missing_periods
            ),
            "status": status,
            "intervals": intervals,
            "warnings": warnings,
        }
