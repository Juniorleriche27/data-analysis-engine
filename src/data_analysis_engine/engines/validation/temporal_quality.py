from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_tolerance_ratio(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'tolerance_ratio' must be numeric."
        )

    tolerance = float(value)

    if (
        not np.isfinite(tolerance)
        or tolerance < 0.0
        or tolerance >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'tolerance_ratio' must satisfy "
            "0 <= tolerance_ratio < 1."
        )

    return tolerance


def _timestamp_from_ns(
    value: int,
) -> str:
    return pd.Timestamp(
        value,
        unit="ns",
        tz="UTC",
    ).isoformat()


class TemporalQualityDiagnostic(
    AnalysisMethod
):
    name = "temporal_quality_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose temporal-axis quality including timestamp "
        "parseability, duplicates, chronological ordering, "
        "interval regularity, and temporal gaps."
    )

    supported_roles = (
        "time",
    )

    minimum_compatible_columns = 1

    selection_mode = "per_column"
    analysis_unit = "column"

    role_requirements = ()

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = kwargs.get("values")

        if values is None:
            raise InvalidAnalysisInputError(
                "'values' is required."
            )

        tolerance_ratio = (
            _validate_tolerance_ratio(
                kwargs.get(
                    "tolerance_ratio",
                    0.05,
                )
            )
        )

        try:
            series = pd.Series(values)
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "'values' must be convertible "
                "to a one-dimensional series."
            ) from exc

        total_count = int(
            len(series)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Temporal-quality diagnostics require "
                "at least one observation."
            )

        missing_mask = series.isna()

        missing_count = int(
            missing_mask.sum()
        )

        observed = series[
            ~missing_mask
        ]

        non_missing_count = int(
            len(observed)
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        if non_missing_count == 0:
            return {
                "diagnostic_type": (
                    "temporal_quality_diagnostic"
                ),
                "total_count": total_count,
                "non_missing_count": 0,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "parseable_count": 0,
                "unparseable_count": 0,
                "parse_success_rate": None,
                "unique_timestamp_count": 0,
                "duplicate_timestamp_count": 0,
                "duplicate_timestamp_rate": None,
                "is_all_missing": True,
                "is_monotonic_increasing": False,
                "is_strictly_increasing": False,
                "out_of_order_count": 0,
                "median_interval_seconds": None,
                "minimum_interval_seconds": None,
                "maximum_interval_seconds": None,
                "irregular_interval_count": 0,
                "irregular_interval_rate": None,
                "gap_count": 0,
                "is_regular": None,
                "first_timestamp": None,
                "last_timestamp": None,
                "tolerance_ratio": tolerance_ratio,
                "risk_level": "critical",
                "quality_status": "all_missing",
                "recommendation": (
                    "No non-missing timestamps are available "
                    "for temporal-quality analysis."
                ),
            }

        try:
            parsed = pd.to_datetime(
                observed,
                errors="coerce",
                utc=True,
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "Temporal values could not be evaluated."
            ) from exc

        parsed_series = pd.Series(
            parsed,
            index=observed.index,
        )

        parseable_mask = (
            ~parsed_series.isna()
        )

        parseable_count = int(
            parseable_mask.sum()
        )

        unparseable_count = int(
            non_missing_count
            - parseable_count
        )

        parse_success_rate = float(
            parseable_count
            / non_missing_count
        )

        valid = parsed_series[
            parseable_mask
        ]

        if parseable_count == 0:
            return {
                "diagnostic_type": (
                    "temporal_quality_diagnostic"
                ),
                "total_count": total_count,
                "non_missing_count": non_missing_count,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "parseable_count": 0,
                "unparseable_count": unparseable_count,
                "parse_success_rate": 0.0,
                "unique_timestamp_count": 0,
                "duplicate_timestamp_count": 0,
                "duplicate_timestamp_rate": 0.0,
                "is_all_missing": False,
                "is_monotonic_increasing": False,
                "is_strictly_increasing": False,
                "out_of_order_count": 0,
                "median_interval_seconds": None,
                "minimum_interval_seconds": None,
                "maximum_interval_seconds": None,
                "irregular_interval_count": 0,
                "irregular_interval_rate": None,
                "gap_count": 0,
                "is_regular": None,
                "first_timestamp": None,
                "last_timestamp": None,
                "tolerance_ratio": tolerance_ratio,
                "risk_level": "critical",
                "quality_status": "unparseable",
                "recommendation": (
                    "No non-missing value could be parsed "
                    "as a timestamp. Correct the temporal "
                    "format before time-series analysis."
                ),
            }

        # pandas may internally store timezone-aware datetimes
        # with units other than nanoseconds (for example us).
        # Timestamp.value guarantees nanoseconds and keeps all
        # downstream interval calculations unit-stable.
        timestamp_ns = np.asarray(
            [
                timestamp.value
                for timestamp
                in valid
            ],
            dtype=np.int64,
        )

        unique_ns = np.unique(
            timestamp_ns
        )

        unique_timestamp_count = int(
            len(unique_ns)
        )

        duplicate_timestamp_count = int(
            parseable_count
            - unique_timestamp_count
        )

        duplicate_timestamp_rate = float(
            duplicate_timestamp_count
            / parseable_count
        )

        if parseable_count >= 2:
            original_intervals_ns = (
                np.diff(timestamp_ns)
            )

            out_of_order_count = int(
                np.sum(
                    original_intervals_ns
                    < 0
                )
            )

            is_monotonic_increasing = bool(
                np.all(
                    original_intervals_ns
                    >= 0
                )
            )

            is_strictly_increasing = bool(
                np.all(
                    original_intervals_ns
                    > 0
                )
            )
        else:
            out_of_order_count = 0
            is_monotonic_increasing = True
            is_strictly_increasing = True

        first_timestamp = _timestamp_from_ns(
            int(
                np.min(unique_ns)
            )
        )

        last_timestamp = _timestamp_from_ns(
            int(
                np.max(unique_ns)
            )
        )

        if unique_timestamp_count >= 2:
            sorted_intervals_seconds = (
                np.diff(unique_ns)
                / 1_000_000_000.0
            )

            median_interval_seconds = float(
                np.median(
                    sorted_intervals_seconds
                )
            )

            minimum_interval_seconds = float(
                np.min(
                    sorted_intervals_seconds
                )
            )

            maximum_interval_seconds = float(
                np.max(
                    sorted_intervals_seconds
                )
            )

            absolute_tolerance = max(
                abs(
                    median_interval_seconds
                )
                * tolerance_ratio,
                1e-9,
            )

            irregular_mask = (
                np.abs(
                    sorted_intervals_seconds
                    - median_interval_seconds
                )
                > absolute_tolerance
            )

            irregular_interval_count = int(
                np.sum(
                    irregular_mask
                )
            )

            interval_count = int(
                len(
                    sorted_intervals_seconds
                )
            )

            irregular_interval_rate = float(
                irregular_interval_count
                / interval_count
            )

            gap_mask = (
                sorted_intervals_seconds
                > (
                    median_interval_seconds
                    * (
                        1.0
                        + tolerance_ratio
                    )
                )
            )

            gap_count = int(
                np.sum(
                    gap_mask
                )
            )

            is_regular = bool(
                irregular_interval_count
                == 0
            )
        else:
            median_interval_seconds = None
            minimum_interval_seconds = None
            maximum_interval_seconds = None
            irregular_interval_count = 0
            irregular_interval_rate = None
            gap_count = 0
            is_regular = None

        if unparseable_count > 0:
            risk_level = "high"
            quality_status = (
                "unparseable_values"
            )
            recommendation = (
                "Some non-missing values could not be "
                "parsed as timestamps. Standardize temporal "
                "formats before time-series analysis."
            )

        elif out_of_order_count > 0:
            risk_level = "high"
            quality_status = (
                "out_of_order"
            )
            recommendation = (
                "Timestamps are not chronologically ordered. "
                "Sort the temporal axis before time-series "
                "analysis."
            )

        elif duplicate_timestamp_count > 0:
            risk_level = "high"
            quality_status = (
                "duplicate_timestamps"
            )
            recommendation = (
                "Duplicate timestamps were detected. Review "
                "whether aggregation or duplicate resolution "
                "is required."
            )

        elif (
            irregular_interval_count > 0
        ):
            risk_level = "moderate"
            quality_status = (
                "irregular_intervals"
            )
            recommendation = (
                "The temporal axis is irregular. Review gaps "
                "or resample the series before methods that "
                "assume regular spacing."
            )

        elif unique_timestamp_count < 2:
            risk_level = "moderate"
            quality_status = (
                "insufficient_temporal_span"
            )
            recommendation = (
                "Only one unique timestamp is available. "
                "Interval regularity cannot be assessed."
            )

        else:
            risk_level = "none"
            quality_status = "clean"
            recommendation = (
                "The temporal axis is parseable, ordered, "
                "unique, and regularly spaced within the "
                "configured tolerance."
            )

        return {
            "diagnostic_type": (
                "temporal_quality_diagnostic"
            ),
            "total_count": total_count,
            "non_missing_count": (
                non_missing_count
            ),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "parseable_count": (
                parseable_count
            ),
            "unparseable_count": (
                unparseable_count
            ),
            "parse_success_rate": (
                parse_success_rate
            ),
            "unique_timestamp_count": (
                unique_timestamp_count
            ),
            "duplicate_timestamp_count": (
                duplicate_timestamp_count
            ),
            "duplicate_timestamp_rate": (
                duplicate_timestamp_rate
            ),
            "is_all_missing": False,
            "is_monotonic_increasing": (
                is_monotonic_increasing
            ),
            "is_strictly_increasing": (
                is_strictly_increasing
            ),
            "out_of_order_count": (
                out_of_order_count
            ),
            "median_interval_seconds": (
                median_interval_seconds
            ),
            "minimum_interval_seconds": (
                minimum_interval_seconds
            ),
            "maximum_interval_seconds": (
                maximum_interval_seconds
            ),
            "irregular_interval_count": (
                irregular_interval_count
            ),
            "irregular_interval_rate": (
                irregular_interval_rate
            ),
            "gap_count": gap_count,
            "is_regular": is_regular,
            "first_timestamp": (
                first_timestamp
            ),
            "last_timestamp": (
                last_timestamp
            ),
            "tolerance_ratio": (
                tolerance_ratio
            ),
            "risk_level": risk_level,
            "quality_status": (
                quality_status
            ),
            "recommendation": (
                recommendation
            ),
        }
