from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_iqr_multiplier(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'iqr_multiplier' must be numeric."
        )

    multiplier = float(value)

    if (
        not np.isfinite(multiplier)
        or multiplier <= 0.0
    ):
        raise InvalidAnalysisInputError(
            "'iqr_multiplier' must be finite and > 0."
        )

    return multiplier


def _outlier_risk_level(
    outlier_rate: float,
) -> str:
    if outlier_rate == 0.0:
        return "none"

    if outlier_rate < 0.01:
        return "low"

    if outlier_rate < 0.05:
        return "moderate"

    if outlier_rate < 0.20:
        return "high"

    return "critical"


def _to_python_scalar(
    value: Any,
) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    return value


class OutlierDiagnostic(
    AnalysisMethod
):
    name = "outlier_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Detect univariate numerical outliers using "
        "Tukey IQR fences while preserving original "
        "row positions and missing-value information."
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

        iqr_multiplier = _validate_iqr_multiplier(
            kwargs.get(
                "iqr_multiplier",
                1.5,
            )
        )

        try:
            series = pd.Series(values)
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "'values' must be convertible "
                "to a one-dimensional series."
            ) from exc

        total_count = int(len(series))

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Outlier diagnostics require "
                "at least one observation."
            )

        missing_mask = np.asarray(
            series.isna(),
            dtype=bool,
        )

        missing_count = int(
            np.sum(missing_mask)
        )

        non_missing_positions = np.flatnonzero(
            ~missing_mask
        )

        non_missing = series[
            ~missing_mask
        ]

        non_missing_count = int(
            len(non_missing)
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        if non_missing_count == 0:
            return {
                "diagnostic_type": (
                    "outlier_diagnostic"
                ),
                "method": "iqr",
                "iqr_multiplier": iqr_multiplier,
                "total_count": total_count,
                "non_missing_count": 0,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "is_all_missing": True,
                "q1": None,
                "median": None,
                "q3": None,
                "iqr": None,
                "lower_fence": None,
                "upper_fence": None,
                "lower_outlier_count": 0,
                "upper_outlier_count": 0,
                "outlier_count": 0,
                "outlier_rate": 0.0,
                "has_outliers": False,
                "outlier_positions": [],
                "outlier_values": [],
                "risk_level": "critical",
                "recommendation": (
                    "No non-missing numerical observations "
                    "are available for outlier analysis."
                ),
            }

        try:
            numeric_values = np.asarray(
                non_missing,
                dtype=float,
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise InvalidAnalysisInputError(
                "Outlier diagnostics require "
                "numeric non-missing values."
            ) from exc

        if not np.all(
            np.isfinite(numeric_values)
        ):
            raise InvalidAnalysisInputError(
                "Outlier diagnostics require finite "
                "non-missing values. Run "
                "'non_finite_value_diagnostic' first."
            )

        q1 = float(
            np.quantile(
                numeric_values,
                0.25,
            )
        )

        median = float(
            np.quantile(
                numeric_values,
                0.50,
            )
        )

        q3 = float(
            np.quantile(
                numeric_values,
                0.75,
            )
        )

        iqr = float(
            q3 - q1
        )

        lower_fence = float(
            q1
            - iqr_multiplier
            * iqr
        )

        upper_fence = float(
            q3
            + iqr_multiplier
            * iqr
        )

        lower_mask = (
            numeric_values
            < lower_fence
        )

        upper_mask = (
            numeric_values
            > upper_fence
        )

        outlier_mask = (
            lower_mask
            | upper_mask
        )

        lower_outlier_count = int(
            np.sum(lower_mask)
        )

        upper_outlier_count = int(
            np.sum(upper_mask)
        )

        outlier_count = int(
            np.sum(outlier_mask)
        )

        outlier_rate = float(
            outlier_count
            / non_missing_count
        )

        outlier_positions = [
            int(position)
            for position
            in non_missing_positions[
                outlier_mask
            ]
        ]

        outlier_values = [
            _to_python_scalar(value)
            for value
            in numeric_values[
                outlier_mask
            ]
        ]

        has_outliers = bool(
            outlier_count > 0
        )

        risk_level = (
            _outlier_risk_level(
                outlier_rate
            )
        )

        if has_outliers:
            recommendation = (
                "Potential outliers were detected using "
                "IQR fences. Review their business and "
                "data context before removing, clipping, "
                "or transforming them."
            )
        else:
            recommendation = (
                "No observations fall outside the "
                "configured IQR fences."
            )

        return {
            "diagnostic_type": (
                "outlier_diagnostic"
            ),
            "method": "iqr",
            "iqr_multiplier": (
                iqr_multiplier
            ),
            "total_count": (
                total_count
            ),
            "non_missing_count": (
                non_missing_count
            ),
            "missing_count": (
                missing_count
            ),
            "missing_rate": (
                missing_rate
            ),
            "is_all_missing": False,
            "q1": q1,
            "median": median,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": (
                lower_fence
            ),
            "upper_fence": (
                upper_fence
            ),
            "lower_outlier_count": (
                lower_outlier_count
            ),
            "upper_outlier_count": (
                upper_outlier_count
            ),
            "outlier_count": (
                outlier_count
            ),
            "outlier_rate": (
                outlier_rate
            ),
            "has_outliers": (
                has_outliers
            ),
            "outlier_positions": (
                outlier_positions
            ),
            "outlier_values": (
                outlier_values
            ),
            "risk_level": (
                risk_level
            ),
            "recommendation": (
                recommendation
            ),
        }
