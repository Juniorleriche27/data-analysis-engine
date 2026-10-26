from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _non_finite_risk_level(
    non_finite_rate: float,
) -> str:
    if non_finite_rate == 0.0:
        return "none"

    if non_finite_rate < 0.01:
        return "low"

    if non_finite_rate < 0.05:
        return "moderate"

    if non_finite_rate < 0.20:
        return "high"

    return "critical"


class NonFiniteValueDiagnostic(
    AnalysisMethod
):
    name = "non_finite_value_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose missing, positive infinite, negative "
        "infinite, and other non-finite values in a "
        "numeric analytical column."
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
        values = kwargs.get(
            "values"
        )

        if values is None:
            raise InvalidAnalysisInputError(
                "'values' is required."
            )

        try:
            series = pd.Series(
                values
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "'values' must be convertible "
                "to a one-dimensional series."
            ) from exc

        total_count = int(
            len(
                series
            )
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Non-finite diagnostics require "
                "at least one observation."
            )

        missing_mask = np.asarray(
            series.isna(),
            dtype=bool,
        )

        missing_count = int(
            np.sum(
                missing_mask
            )
        )

        non_missing = series[
            ~missing_mask
        ]

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
                "Non-finite diagnostics require "
                "numeric non-missing values."
            ) from exc

        positive_infinity_count = int(
            np.sum(
                np.isposinf(
                    numeric_values
                )
            )
        )

        negative_infinity_count = int(
            np.sum(
                np.isneginf(
                    numeric_values
                )
            )
        )

        infinity_count = int(
            positive_infinity_count
            + negative_infinity_count
        )

        finite_non_missing_count = int(
            np.sum(
                np.isfinite(
                    numeric_values
                )
            )
        )

        non_finite_count = int(
            missing_count
            + infinity_count
        )

        finite_count = int(
            total_count
            - non_finite_count
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        infinity_rate = float(
            infinity_count
            / total_count
        )

        non_finite_rate = float(
            non_finite_count
            / total_count
        )

        finite_rate = float(
            finite_count
            / total_count
        )

        positive_infinity_positions = [
            int(
                position
            )
            for position, value
            in enumerate(
                series
            )
            if (
                not pd.isna(
                    value
                )
                and np.isposinf(
                    float(
                        value
                    )
                )
            )
        ]

        negative_infinity_positions = [
            int(
                position
            )
            for position, value
            in enumerate(
                series
            )
            if (
                not pd.isna(
                    value
                )
                and np.isneginf(
                    float(
                        value
                    )
                )
            )
        ]

        missing_positions = [
            int(
                position
            )
            for position
            in np.flatnonzero(
                missing_mask
            )
        ]

        has_infinity = bool(
            infinity_count
            > 0
        )

        has_non_finite = bool(
            non_finite_count
            > 0
        )

        all_non_finite = bool(
            non_finite_count
            == total_count
        )

        risk_level = (
            _non_finite_risk_level(
                non_finite_rate
            )
        )

        if all_non_finite:
            recommendation = (
                "No finite observations are available. "
                "The column should not be used for "
                "numerical analysis until corrected."
            )

        elif has_infinity:
            recommendation = (
                "Infinite values were detected. "
                "Review their origin before statistical "
                "analysis, regression, or modelling."
            )

        elif missing_count > 0:
            recommendation = (
                "Missing values were detected but no "
                "infinite values were found. Apply an "
                "appropriate missing-data strategy."
            )

        else:
            recommendation = (
                "All observations are finite."
            )

        return {
            "diagnostic_type": (
                "non_finite_value_diagnostic"
            ),
            "total_count": (
                total_count
            ),
            "finite_count": (
                finite_count
            ),
            "finite_non_missing_count": (
                finite_non_missing_count
            ),
            "missing_count": (
                missing_count
            ),
            "positive_infinity_count": (
                positive_infinity_count
            ),
            "negative_infinity_count": (
                negative_infinity_count
            ),
            "infinity_count": (
                infinity_count
            ),
            "non_finite_count": (
                non_finite_count
            ),
            "finite_rate": (
                finite_rate
            ),
            "missing_rate": (
                missing_rate
            ),
            "infinity_rate": (
                infinity_rate
            ),
            "non_finite_rate": (
                non_finite_rate
            ),
            "missing_positions": (
                missing_positions
            ),
            "positive_infinity_positions": (
                positive_infinity_positions
            ),
            "negative_infinity_positions": (
                negative_infinity_positions
            ),
            "has_infinity": (
                has_infinity
            ),
            "has_non_finite": (
                has_non_finite
            ),
            "all_non_finite": (
                all_non_finite
            ),
            "risk_level": (
                risk_level
            ),
            "risk_basis": (
                "non_finite_rate"
            ),
            "risk_thresholds": {
                "low": "<1%",
                "moderate": "1%-<5%",
                "high": "5%-<20%",
                "critical": ">=20%",
            },
            "recommendation": (
                recommendation
            ),
        }
