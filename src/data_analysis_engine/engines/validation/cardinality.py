from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_ratio_threshold(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'high_cardinality_ratio' must be numeric."
        )

    threshold = float(value)

    if (
        not np.isfinite(threshold)
        or threshold <= 0.0
        or threshold > 1.0
    ):
        raise InvalidAnalysisInputError(
            "'high_cardinality_ratio' must satisfy "
            "0 < high_cardinality_ratio <= 1."
        )

    return threshold


def _validate_minimum_unique_count(
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise InvalidAnalysisInputError(
            "'minimum_unique_count' must be an integer."
        )

    count = int(value)

    if count < 2:
        raise InvalidAnalysisInputError(
            "'minimum_unique_count' must be >= 2."
        )

    return count


def _to_python_scalar(
    value: Any,
) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    return value


class CardinalityDiagnostic(
    AnalysisMethod
):
    name = "cardinality_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose categorical cardinality, uniqueness, "
        "singleton values, and dominant-frequency structure."
    )

    supported_roles = (
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
        values = kwargs.get("values")

        if values is None:
            raise InvalidAnalysisInputError(
                "'values' is required."
            )

        high_cardinality_ratio = (
            _validate_ratio_threshold(
                kwargs.get(
                    "high_cardinality_ratio",
                    0.90,
                )
            )
        )

        minimum_unique_count = (
            _validate_minimum_unique_count(
                kwargs.get(
                    "minimum_unique_count",
                    20,
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

        total_count = int(len(series))

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Cardinality diagnostics require "
                "at least one observation."
            )

        missing_count = int(
            series.isna().sum()
        )

        observed = series.dropna()

        non_missing_count = int(
            len(observed)
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        is_all_missing = bool(
            non_missing_count == 0
        )

        if is_all_missing:
            return {
                "diagnostic_type": (
                    "cardinality_diagnostic"
                ),
                "total_count": total_count,
                "non_missing_count": 0,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "unique_count": 0,
                "unique_ratio": None,
                "singleton_count": 0,
                "singleton_ratio": None,
                "dominant_value": None,
                "dominant_count": 0,
                "dominant_ratio": None,
                "is_all_missing": True,
                "is_constant": False,
                "is_unique": False,
                "is_high_cardinality": False,
                "cardinality_status": "all_missing",
                "risk_level": "critical",
                "high_cardinality_ratio": (
                    high_cardinality_ratio
                ),
                "minimum_unique_count": (
                    minimum_unique_count
                ),
                "recommendation": (
                    "No non-missing categorical observations "
                    "are available for cardinality analysis."
                ),
            }

        try:
            counts = observed.value_counts(
                dropna=False
            )
        except TypeError as exc:
            raise InvalidAnalysisInputError(
                "Cardinality diagnostics require "
                "hashable scalar observations."
            ) from exc

        unique_count = int(len(counts))

        unique_ratio = float(
            unique_count
            / non_missing_count
        )

        singleton_count = int(
            np.sum(
                np.asarray(
                    counts == 1,
                    dtype=bool,
                )
            )
        )

        singleton_ratio = float(
            singleton_count
            / non_missing_count
        )

        dominant_value = _to_python_scalar(
            counts.index[0]
        )

        dominant_count = int(
            counts.iloc[0]
        )

        dominant_ratio = float(
            dominant_count
            / non_missing_count
        )

        is_constant = bool(
            unique_count == 1
        )

        is_unique = bool(
            unique_count
            == non_missing_count
        )

        is_high_cardinality = bool(
            not is_unique
            and unique_count
            >= minimum_unique_count
            and unique_ratio
            >= high_cardinality_ratio
        )

        if is_constant:
            cardinality_status = "constant"
            risk_level = "critical"
            recommendation = (
                "The column contains only one distinct "
                "non-missing value and carries no categorical "
                "variation."
            )

        elif is_unique:
            cardinality_status = "unique"
            risk_level = "critical"
            recommendation = (
                "Every non-missing observation is unique. "
                "Review whether the column behaves like an "
                "identifier before categorical encoding."
            )

        elif is_high_cardinality:
            cardinality_status = "high"
            risk_level = "high"
            recommendation = (
                "High categorical cardinality was detected. "
                "Review rare categories and encoding strategy "
                "before modelling."
            )

        else:
            cardinality_status = "normal"
            risk_level = "none"
            recommendation = (
                "No problematic categorical cardinality "
                "pattern was detected."
            )

        return {
            "diagnostic_type": (
                "cardinality_diagnostic"
            ),
            "total_count": total_count,
            "non_missing_count": (
                non_missing_count
            ),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "unique_count": unique_count,
            "unique_ratio": unique_ratio,
            "singleton_count": (
                singleton_count
            ),
            "singleton_ratio": (
                singleton_ratio
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
            "is_all_missing": False,
            "is_constant": is_constant,
            "is_unique": is_unique,
            "is_high_cardinality": (
                is_high_cardinality
            ),
            "cardinality_status": (
                cardinality_status
            ),
            "risk_level": risk_level,
            "high_cardinality_ratio": (
                high_cardinality_ratio
            ),
            "minimum_unique_count": (
                minimum_unique_count
            ),
            "recommendation": (
                recommendation
            ),
        }
