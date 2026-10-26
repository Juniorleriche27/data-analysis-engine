from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_relative_variance_threshold(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'relative_variance_threshold' must be numeric."
        )

    threshold = float(value)

    if (
        not np.isfinite(threshold)
        or threshold < 0.0
        or threshold >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'relative_variance_threshold' must satisfy "
            "0 <= relative_variance_threshold < 1."
        )

    return threshold


class VarianceDiagnostic(
    AnalysisMethod
):
    name = "variance_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose zero and near-zero numerical variance "
        "using a scale-invariant relative-variance measure."
    )

    supported_roles = (
        "measure",
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

        threshold = (
            _validate_relative_variance_threshold(
                kwargs.get(
                    "relative_variance_threshold",
                    1e-6,
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
                "Variance diagnostics require "
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
                    "variance_diagnostic"
                ),
                "total_count": total_count,
                "non_missing_count": 0,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "is_all_missing": True,
                "mean": None,
                "variance": None,
                "standard_deviation": None,
                "minimum": None,
                "maximum": None,
                "range": None,
                "mean_square": None,
                "relative_variance": None,
                "relative_variance_threshold": threshold,
                "is_zero_variance": False,
                "is_near_zero_variance": False,
                "variance_status": "all_missing",
                "risk_level": "critical",
                "recommendation": (
                    "No non-missing numerical observations "
                    "are available for variance analysis."
                ),
            }

        try:
            numeric_values = np.asarray(
                observed,
                dtype=float,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise InvalidAnalysisInputError(
                "Variance diagnostics require "
                "numeric non-missing values."
            ) from exc

        if not np.all(
            np.isfinite(numeric_values)
        ):
            raise InvalidAnalysisInputError(
                "Variance diagnostics require finite "
                "non-missing values. Run "
                "'non_finite_value_diagnostic' first."
            )

        mean = float(
            np.mean(numeric_values)
        )

        variance = float(
            np.var(
                numeric_values,
                ddof=0,
            )
        )

        standard_deviation = float(
            np.sqrt(variance)
        )

        minimum = float(
            np.min(numeric_values)
        )

        maximum = float(
            np.max(numeric_values)
        )

        value_range = float(
            maximum - minimum
        )

        mean_square = float(
            np.mean(
                np.square(
                    numeric_values
                )
            )
        )

        if mean_square == 0.0:
            relative_variance = 0.0
        else:
            relative_variance = float(
                variance
                / mean_square
            )

        is_zero_variance = bool(
            variance == 0.0
        )

        is_near_zero_variance = bool(
            not is_zero_variance
            and relative_variance
            <= threshold
        )

        if is_zero_variance:
            variance_status = "zero_variance"
            risk_level = "critical"
            recommendation = (
                "The column has zero variance and carries "
                "no numerical variation. Exclude it from "
                "variance-dependent modelling."
            )

        elif is_near_zero_variance:
            variance_status = "near_zero_variance"
            risk_level = "high"
            recommendation = (
                "The column has extremely low variation "
                "relative to its numerical scale. Review "
                "its usefulness before modelling."
            )

        else:
            variance_status = "adequate_variance"
            risk_level = "none"
            recommendation = (
                "No zero or near-zero variance problem "
                "was detected."
            )

        return {
            "diagnostic_type": (
                "variance_diagnostic"
            ),
            "total_count": total_count,
            "non_missing_count": (
                non_missing_count
            ),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "is_all_missing": False,
            "mean": mean,
            "variance": variance,
            "standard_deviation": (
                standard_deviation
            ),
            "minimum": minimum,
            "maximum": maximum,
            "range": value_range,
            "mean_square": mean_square,
            "relative_variance": (
                relative_variance
            ),
            "relative_variance_threshold": (
                threshold
            ),
            "is_zero_variance": (
                is_zero_variance
            ),
            "is_near_zero_variance": (
                is_near_zero_variance
            ),
            "variance_status": (
                variance_status
            ),
            "risk_level": risk_level,
            "recommendation": (
                recommendation
            ),
        }
