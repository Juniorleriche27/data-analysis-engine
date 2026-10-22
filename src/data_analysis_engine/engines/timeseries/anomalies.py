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


MAD_NORMALIZATION = 0.6744897501960817


def _validate_window(
    value: Any,
    series_length: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise InvalidAnalysisInputError(
            "'window' must be an integer."
        )

    if value < 2:
        raise InvalidAnalysisInputError(
            "'window' must be at least 2."
        )

    if value >= series_length:
        raise InvalidAnalysisInputError(
            "'window' must be smaller than "
            "the series length."
        )

    return value


def _validate_threshold(
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
            "'threshold' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        raise InvalidAnalysisInputError(
            "'threshold' must be finite and "
            "greater than zero."
        )

    return value


class _TemporalAnomalyMethod(AnalysisMethod):
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


class RollingZScoreAnomaly(
    _TemporalAnomalyMethod
):
    name = "rolling_zscore_anomaly"
    version = "1.0"

    description = (
        "Detect point anomalies using a rolling z-score "
        "computed strictly from previous observations."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        window = _validate_window(
            kwargs.get(
                "window",
                5,
            ),
            len(values),
        )

        threshold = _validate_threshold(
            kwargs.get(
                "threshold",
                3.0,
            )
        )

        records: list[
            dict[str, Any]
        ] = []

        anomaly_count = 0

        for index in range(
            window,
            len(values),
        ):
            baseline = values[
                index - window:
                index
            ]

            current = float(
                values[index]
            )

            baseline_mean = float(
                np.mean(
                    baseline
                )
            )

            baseline_std = float(
                np.std(
                    baseline,
                    ddof=1,
                )
            )

            if baseline_std > 0:
                z_score = float(
                    (
                        current
                        - baseline_mean
                    )
                    / baseline_std
                )

                score_defined = True

                is_anomaly = (
                    abs(z_score)
                    >= threshold
                )

                reason = (
                    "threshold_exceeded"
                    if is_anomaly
                    else "within_threshold"
                )

            else:
                z_score = None
                score_defined = False

                is_anomaly = (
                    current
                    != baseline_mean
                )

                reason = (
                    "departure_from_zero_variance_baseline"
                    if is_anomaly
                    else "matches_zero_variance_baseline"
                )

            if is_anomaly:
                anomaly_count += 1

            records.append(
                {
                    "timestamp": _serialize_time(
                        time[index]
                    ),
                    "index": index,
                    "value": current,
                    "baseline_start_index": (
                        index - window
                    ),
                    "baseline_end_index": (
                        index - 1
                    ),
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "z_score": z_score,
                    "score_defined": (
                        score_defined
                    ),
                    "threshold": threshold,
                    "anomaly": is_anomaly,
                    "reason": reason,
                }
            )

        return {
            "method": "rolling_zscore",
            "sample_size": len(values),
            "window": window,
            "threshold": threshold,
            "baseline_policy": (
                "previous_observations_only"
            ),
            "evaluated_observations": len(
                records
            ),
            "anomaly_count": anomaly_count,
            "records": records,
        }


class RollingMADAnomaly(
    _TemporalAnomalyMethod
):
    name = "rolling_mad_anomaly"
    version = "1.0"

    description = (
        "Detect point anomalies using a rolling robust "
        "modified z-score based on median absolute deviation "
        "from previous observations."
    )

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        time, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        window = _validate_window(
            kwargs.get(
                "window",
                5,
            ),
            len(values),
        )

        threshold = _validate_threshold(
            kwargs.get(
                "threshold",
                3.5,
            )
        )

        records: list[
            dict[str, Any]
        ] = []

        anomaly_count = 0

        for index in range(
            window,
            len(values),
        ):
            baseline = values[
                index - window:
                index
            ]

            current = float(
                values[index]
            )

            baseline_median = float(
                np.median(
                    baseline
                )
            )

            absolute_deviations = np.abs(
                baseline
                - baseline_median
            )

            mad = float(
                np.median(
                    absolute_deviations
                )
            )

            if mad > 0:
                modified_z_score = float(
                    MAD_NORMALIZATION
                    * (
                        current
                        - baseline_median
                    )
                    / mad
                )

                score_defined = True

                is_anomaly = (
                    abs(
                        modified_z_score
                    )
                    >= threshold
                )

                reason = (
                    "threshold_exceeded"
                    if is_anomaly
                    else "within_threshold"
                )

            else:
                modified_z_score = None
                score_defined = False

                is_anomaly = (
                    current
                    != baseline_median
                )

                reason = (
                    "departure_from_zero_mad_baseline"
                    if is_anomaly
                    else "matches_zero_mad_baseline"
                )

            if is_anomaly:
                anomaly_count += 1

            records.append(
                {
                    "timestamp": _serialize_time(
                        time[index]
                    ),
                    "index": index,
                    "value": current,
                    "baseline_start_index": (
                        index - window
                    ),
                    "baseline_end_index": (
                        index - 1
                    ),
                    "baseline_median": (
                        baseline_median
                    ),
                    "baseline_mad": mad,
                    "modified_z_score": (
                        modified_z_score
                    ),
                    "score_defined": (
                        score_defined
                    ),
                    "threshold": threshold,
                    "anomaly": is_anomaly,
                    "reason": reason,
                }
            )

        return {
            "method": "rolling_mad",
            "sample_size": len(values),
            "window": window,
            "threshold": threshold,
            "baseline_policy": (
                "previous_observations_only"
            ),
            "evaluated_observations": len(
                records
            ),
            "anomaly_count": anomaly_count,
            "records": records,
        }
