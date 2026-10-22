from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.basic import (
    _validate_time_series,
)
from data_analysis_engine.engines.timeseries.trend import (
    _fit_linear_trend,
)


def _positive_threshold(
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


class TimeSeriesStabilityDiagnostic(AnalysisMethod):
    name = "time_series_stability_diagnostic"
    category = "timeseries"
    version = "1.0"

    description = (
        "Provide descriptive stability diagnostics for an "
        "ordered numerical series. This is not a formal "
        "stationarity test such as ADF or KPSS."
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
        _, values = _validate_time_series(
            kwargs.get("time"),
            kwargs.get("values"),
        )

        if len(values) < 6:
            raise InvalidAnalysisInputError(
                "Stability diagnostics require at least "
                "6 observations."
            )

        mean_shift_threshold = _positive_threshold(
            kwargs.get(
                "max_standardized_mean_shift",
                0.5,
            ),
            "max_standardized_mean_shift",
        )

        variance_ratio_threshold = _positive_threshold(
            kwargs.get(
                "max_variance_ratio",
                2.0,
            ),
            "max_variance_ratio",
        )

        trend_threshold = _positive_threshold(
            kwargs.get(
                "max_standardized_trend_strength",
                1.0,
            ),
            "max_standardized_trend_strength",
        )

        autocorrelation_threshold = _positive_threshold(
            kwargs.get(
                "max_abs_lag1_autocorrelation",
                0.8,
            ),
            "max_abs_lag1_autocorrelation",
        )

        n = len(values)
        half_size = n // 2

        first_half = values[
            :half_size
        ]

        second_half = values[
            -half_size:
        ]

        overall_mean = float(
            np.mean(values)
        )

        overall_std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        first_mean = float(
            np.mean(first_half)
        )

        second_mean = float(
            np.mean(second_half)
        )

        absolute_mean_shift = abs(
            second_mean
            - first_mean
        )

        if overall_std > 0:
            standardized_mean_shift = float(
                absolute_mean_shift
                / overall_std
            )
        else:
            standardized_mean_shift = 0.0

        first_variance = float(
            np.var(
                first_half,
                ddof=1,
            )
        )

        second_variance = float(
            np.var(
                second_half,
                ddof=1,
            )
        )

        minimum_variance = min(
            first_variance,
            second_variance,
        )

        maximum_variance = max(
            first_variance,
            second_variance,
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
            variance_ratio = None
            variance_ratio_defined = False

        trend_fit = _fit_linear_trend(
            values
        )

        slope = float(
            trend_fit["slope"]
        )

        if overall_std > 0:
            standardized_trend_strength = float(
                abs(slope)
                * (n - 1)
                / overall_std
            )
        else:
            standardized_trend_strength = 0.0

        left = values[:-1]
        right = values[1:]

        lag1_defined = not (
            np.all(
                left == left[0]
            )
            or np.all(
                right == right[0]
            )
        )

        lag1_autocorrelation = (
            float(
                np.corrcoef(
                    left,
                    right,
                )[0, 1]
            )
            if lag1_defined
            else None
        )

        checks: list[
            dict[str, Any]
        ] = []

        checks.append(
            {
                "name": "mean_stability",
                "evaluated": True,
                "passed": (
                    standardized_mean_shift
                    <= mean_shift_threshold
                ),
                "value": standardized_mean_shift,
                "threshold": mean_shift_threshold,
            }
        )

        checks.append(
            {
                "name": "variance_stability",
                "evaluated": True,
                "passed": (
                    variance_ratio
                    <= variance_ratio_threshold
                    if variance_ratio_defined
                    else False
                ),
                "value": variance_ratio,
                "threshold": variance_ratio_threshold,
            }
        )

        checks.append(
            {
                "name": "trend_stability",
                "evaluated": True,
                "passed": (
                    standardized_trend_strength
                    <= trend_threshold
                ),
                "value": standardized_trend_strength,
                "threshold": trend_threshold,
            }
        )

        checks.append(
            {
                "name": "lag1_autocorrelation",
                "evaluated": lag1_defined,
                "passed": (
                    abs(
                        lag1_autocorrelation
                    )
                    <= autocorrelation_threshold
                    if lag1_defined
                    else None
                ),
                "value": lag1_autocorrelation,
                "threshold": autocorrelation_threshold,
            }
        )

        instability_signals = [
            check["name"]
            for check in checks
            if (
                check["evaluated"]
                and check["passed"] is False
            )
        ]

        evaluated_count = sum(
            1
            for check in checks
            if check["evaluated"]
        )

        passed_count = sum(
            1
            for check in checks
            if (
                check["evaluated"]
                and check["passed"] is True
            )
        )

        assessment = (
            "no_instability_signal"
            if not instability_signals
            else "instability_signals_detected"
        )

        return {
            "diagnostic_type": "descriptive",
            "formal_stationarity_test": False,
            "sample_size": n,
            "assessment": assessment,
            "instability_signals": instability_signals,
            "evaluated_checks": evaluated_count,
            "passed_checks": passed_count,
            "metrics": {
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "first_half_mean": first_mean,
                "second_half_mean": second_mean,
                "standardized_mean_shift": (
                    standardized_mean_shift
                ),
                "first_half_variance": first_variance,
                "second_half_variance": second_variance,
                "variance_ratio": variance_ratio,
                "variance_ratio_defined": (
                    variance_ratio_defined
                ),
                "linear_slope": slope,
                "standardized_trend_strength": (
                    standardized_trend_strength
                ),
                "lag1_autocorrelation": (
                    lag1_autocorrelation
                ),
                "lag1_autocorrelation_defined": (
                    lag1_defined
                ),
            },
            "thresholds": {
                "max_standardized_mean_shift": (
                    mean_shift_threshold
                ),
                "max_variance_ratio": (
                    variance_ratio_threshold
                ),
                "max_standardized_trend_strength": (
                    trend_threshold
                ),
                "max_abs_lag1_autocorrelation": (
                    autocorrelation_threshold
                ),
            },
            "checks": checks,
            "limitations": [
                (
                    "This diagnostic is descriptive and "
                    "does not replace ADF, KPSS, or another "
                    "formal stationarity test."
                )
            ],
        }
