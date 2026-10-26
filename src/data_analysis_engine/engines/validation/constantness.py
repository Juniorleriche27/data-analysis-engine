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
        or threshold >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'dominance_threshold' must satisfy "
            "0 < dominance_threshold < 1."
        )

    return threshold


def _to_python_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

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

        dominance_threshold = (
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

        missing_count = int(
            series.isna().sum()
        )

        observed = series.dropna()

        non_missing_count = int(
            len(
                observed
            )
        )

        # Alias retained for the newer Data Quality contract.
        observed_count = (
            non_missing_count
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        is_all_missing = bool(
            non_missing_count
            == 0
        )

        if is_all_missing:
            return {
                "diagnostic_type": (
                    "constant_column_diagnostic"
                ),
                "total_count": (
                    total_count
                ),
                "non_missing_count": 0,
                "observed_count": 0,
                "missing_count": (
                    missing_count
                ),
                "missing_rate": (
                    missing_rate
                ),
                "missing_percentage": float(
                    missing_rate
                    * 100.0
                ),
                "unique_count": 0,
                "dominant_value": None,
                "dominant_count": 0,
                "dominant_ratio": None,
                "dominant_percentage": None,
                "dominance_threshold": (
                    dominance_threshold
                ),
                "is_constant": False,
                "is_quasi_constant": False,
                "is_near_constant": False,
                "is_all_missing": True,
                "all_missing": True,
                "risk_level": "critical",
                "variability_status": (
                    "all_missing"
                ),
                "informative_variation": False,
                "missing_values_ignored": True,
            }

        try:
            counts = observed.value_counts(
                dropna=False
            )
        except TypeError as exc:
            raise InvalidAnalysisInputError(
                "Constant-column diagnostics require "
                "hashable scalar observations."
            ) from exc

        unique_count = int(
            len(
                counts
            )
        )

        dominant_value = (
            _to_python_scalar(
                counts.index[
                    0
                ]
            )
        )

        dominant_count = int(
            counts.iloc[
                0
            ]
        )

        dominant_ratio = float(
            dominant_count
            / non_missing_count
        )

        is_constant = bool(
            unique_count
            == 1
        )

        is_quasi_constant = bool(
            not is_constant
            and dominant_ratio
            >= dominance_threshold
        )

        # Synonym introduced by the newer contract.
        is_near_constant = (
            is_quasi_constant
        )

        if is_constant:
            risk_level = "critical"
            variability_status = (
                "constant"
            )

        elif is_quasi_constant:
            risk_level = "high"
            variability_status = (
                "near_constant"
            )

        else:
            risk_level = "none"
            variability_status = (
                "variable"
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
            "observed_count": (
                observed_count
            ),
            "missing_count": (
                missing_count
            ),
            "missing_rate": (
                missing_rate
            ),
            "missing_percentage": float(
                missing_rate
                * 100.0
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
            "dominant_percentage": float(
                dominant_ratio
                * 100.0
            ),
            "dominance_threshold": (
                dominance_threshold
            ),
            "is_constant": (
                is_constant
            ),
            "is_quasi_constant": (
                is_quasi_constant
            ),
            "is_near_constant": (
                is_near_constant
            ),
            "is_all_missing": False,
            "all_missing": False,
            "risk_level": (
                risk_level
            ),
            "variability_status": (
                variability_status
            ),
            "informative_variation": bool(
                not is_constant
                and not is_quasi_constant
            ),
            "missing_values_ignored": True,
        }
