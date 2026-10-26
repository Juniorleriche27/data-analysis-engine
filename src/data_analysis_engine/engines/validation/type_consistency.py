from collections import Counter
from datetime import date, datetime
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _is_numeric_string(
    value: str,
) -> bool:
    stripped = value.strip()

    if not stripped:
        return False

    try:
        parsed = float(stripped)
    except ValueError:
        return False

    return bool(
        np.isfinite(parsed)
    )


def _type_family(
    value: Any,
) -> str:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return "boolean"

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
            date,
            np.datetime64,
        ),
    ):
        return "datetime"

    if isinstance(
        value,
        Real,
    ):
        return "numeric"

    if isinstance(
        value,
        str,
    ):
        if _is_numeric_string(value):
            return "numeric_string"

        return "string"

    return "other"


def _recommended_target_type(
    families: set[str],
) -> str | None:
    if not families:
        return None

    if families <= {
        "numeric",
        "numeric_string",
    }:
        return "numeric"

    if families == {"boolean"}:
        return "boolean"

    if families == {"datetime"}:
        return "datetime"

    if families == {"string"}:
        return "string"

    if families == {"other"}:
        return "other"

    return None


class TypeConsistencyDiagnostic(
    AnalysisMethod
):
    name = "type_consistency_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose heterogeneous value types, numeric values "
        "stored as strings, and coercion-compatible type "
        "patterns within an analytical column."
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
        values = kwargs.get("values")

        if values is None:
            raise InvalidAnalysisInputError(
                "'values' is required."
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
                "Type-consistency diagnostics require "
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
                    "type_consistency_diagnostic"
                ),
                "total_count": total_count,
                "non_missing_count": 0,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "is_all_missing": True,
                "type_families": [],
                "type_family_counts": {},
                "dominant_type_family": None,
                "dominant_type_count": 0,
                "dominant_type_ratio": None,
                "numeric_string_count": 0,
                "numeric_string_ratio": None,
                "is_homogeneous": False,
                "is_numeric_coercible": False,
                "has_mixed_types": False,
                "has_numeric_strings": False,
                "consistency_status": "all_missing",
                "recommended_target_type": None,
                "risk_level": "critical",
                "recommendation": (
                    "No non-missing values are available "
                    "for type-consistency analysis."
                ),
            }

        families = [
            _type_family(value)
            for value
            in observed
        ]

        counts = Counter(
            families
        )

        family_set = set(
            counts
        )

        ordered_families = sorted(
            family_set
        )

        dominant_type_family, dominant_type_count = (
            counts.most_common(1)[0]
        )

        dominant_type_count = int(
            dominant_type_count
        )

        dominant_type_ratio = float(
            dominant_type_count
            / non_missing_count
        )

        numeric_string_count = int(
            counts.get(
                "numeric_string",
                0,
            )
        )

        numeric_string_ratio = float(
            numeric_string_count
            / non_missing_count
        )

        has_numeric_strings = bool(
            numeric_string_count > 0
        )

        is_homogeneous = bool(
            len(family_set) == 1
        )

        is_numeric_coercible = bool(
            family_set <= {
                "numeric",
                "numeric_string",
            }
        )

        has_mixed_types = bool(
            len(family_set) > 1
        )

        target_type = (
            _recommended_target_type(
                family_set
            )
        )

        if (
            has_mixed_types
            and not is_numeric_coercible
        ):
            consistency_status = (
                "incompatible_mixed_types"
            )
            risk_level = "high"
            recommendation = (
                "Incompatible value-type families were "
                "detected. Standardize or explicitly convert "
                "the column before analytical use."
            )

        elif has_numeric_strings:
            if has_mixed_types:
                consistency_status = (
                    "numeric_coercible_mixed"
                )
            else:
                consistency_status = (
                    "numeric_stored_as_text"
                )

            risk_level = "moderate"
            recommendation = (
                "Numeric-looking values are stored as text. "
                "Convert the column to a numerical dtype "
                "before numerical analysis."
            )

        else:
            consistency_status = "consistent"
            risk_level = "none"
            recommendation = (
                "No problematic value-type inconsistency "
                "was detected."
            )

        return {
            "diagnostic_type": (
                "type_consistency_diagnostic"
            ),
            "total_count": total_count,
            "non_missing_count": (
                non_missing_count
            ),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "is_all_missing": False,
            "type_families": (
                ordered_families
            ),
            "type_family_counts": {
                family: int(
                    counts[family]
                )
                for family
                in ordered_families
            },
            "dominant_type_family": (
                dominant_type_family
            ),
            "dominant_type_count": (
                dominant_type_count
            ),
            "dominant_type_ratio": (
                dominant_type_ratio
            ),
            "numeric_string_count": (
                numeric_string_count
            ),
            "numeric_string_ratio": (
                numeric_string_ratio
            ),
            "is_homogeneous": (
                is_homogeneous
            ),
            "is_numeric_coercible": (
                is_numeric_coercible
            ),
            "has_mixed_types": (
                has_mixed_types
            ),
            "has_numeric_strings": (
                has_numeric_strings
            ),
            "consistency_status": (
                consistency_status
            ),
            "recommended_target_type": (
                target_type
            ),
            "risk_level": (
                risk_level
            ),
            "recommendation": (
                recommendation
            ),
        }
