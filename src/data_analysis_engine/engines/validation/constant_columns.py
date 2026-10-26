from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_dominance_threshold(
    value: Any,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise InvalidAnalysisInputError(
            "'dominance_threshold' must be numeric."
        )

    threshold = float(
        value
    )

    if (
        not np.isfinite(
            threshold
        )
        or threshold <= 0.0
        or threshold > 1.0
    ):
        raise InvalidAnalysisInputError(
            "'dominance_threshold' must satisfy "
            "0 < dominance_threshold <= 1."
        )

    return threshold


def _to_python_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        pd.Timedelta,
    ):
        return str(
            value
        )

    return value


class ConstantColumnDiagnostic(
    AnalysisMethod
):
    name = "constant_column_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Detect constant and quasi-constant columns "
        "using the dominance ratio among non-missing "
        "observations."
    )

    supported_roles = (
        "time",
        "measure",
        "dimension",
        "text",
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

        threshold = (
            _validate_dominance_threshold(
                kwargs.get(
                    "dominance_threshold",
                    0.95,
                )
            )
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
                "Constant-column diagnostics require "
                "at least one observation."
            )

        missing_mask = (
            series.isna()
        )

        missing_count = int(
            missing_mask.sum()
        )

        non_missing = (
            series[
                ~missing_mask
            ]
        )

        non_missing_count = int(
            len(
                non_missing
            )
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        unique_count = int(
            non_missing.nunique(
                dropna=True
            )
        )

        all_missing = bool(
            non_missing_count
            == 0
        )

        if all_missing:
            dominant_value = None
            dominant_count = 0
            dominant_ratio = None
            dominant_ratio_total = 0.0

        else:
            counts = (
                non_missing.value_counts(
                    dropna=False,
                    sort=False,
                )
            )

            dominant_label = (
                counts.idxmax()
            )

            dominant_count = int(
                counts.loc[
                    dominant_label
                ]
            )

            dominant_value = (
                _to_python_value(
                    dominant_label
                )
            )

            dominant_ratio = float(
                dominant_count
                / non_missing_count
            )

            dominant_ratio_total = float(
                dominant_count
                / total_count
            )

        is_constant = bool(
            not all_missing
            and unique_count == 1
        )

        is_quasi_constant = bool(
            not all_missing
            and not is_constant
            and dominant_ratio is not None
            and dominant_ratio
            >= threshold
        )

        if all_missing:
            risk_level = "critical"
            recommendation = (
                "No observed values are available. "
                "The column should not be used for "
                "analytical modelling."
            )

        elif is_constant:
            risk_level = "critical"
            recommendation = (
                "The column contains no observed "
                "variation and should generally be "
                "excluded from variance-based analysis "
                "and predictive modelling."
            )

        elif is_quasi_constant:
            risk_level = "high"
            recommendation = (
                "The column is strongly dominated by "
                "one value. Review its analytical value "
                "before modelling."
            )

        else:
            risk_level = "none"
            recommendation = (
                "No constant or quasi-constant issue "
                "was detected at the configured threshold."
            )

        return {
            "diagnostic_type": (
                "constant_column_diagnostic"
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
            "unique_count": (
                unique_count
            ),
            "dominant_value": (
                dominant_value
            ),
            "dominant_count": (
                dominant_count
            ),
            "dominant_ratio": (
                dominant_ratio
            ),
            "dominant_ratio_total": (
                dominant_ratio_total
            ),
            "dominance_threshold": (
                threshold
            ),
            "is_all_missing": (
                all_missing
            ),
            "is_constant": (
                is_constant
            ),
            "is_quasi_constant": (
                is_quasi_constant
            ),
            "has_variation": bool(
                unique_count
                > 1
            ),
            "risk_level": (
                risk_level
            ),
            "recommendation": (
                recommendation
            ),
        }
