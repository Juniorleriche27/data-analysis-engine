from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _longest_true_run(
    mask: np.ndarray,
) -> int:
    longest = 0
    current = 0

    for value in mask:
        if bool(value):
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return int(
        longest
    )


def _count_true_runs(
    mask: np.ndarray,
) -> int:
    if len(mask) == 0:
        return 0

    runs = 0
    previous = False

    for value in mask:
        current = bool(
            value
        )

        if (
            current
            and not previous
        ):
            runs += 1

        previous = current

    return int(
        runs
    )


def _missingness_severity(
    missing_rate: float,
) -> str:
    if missing_rate == 0.0:
        return "none"

    if missing_rate < 0.05:
        return "low"

    if missing_rate < 0.20:
        return "moderate"

    if missing_rate < 0.50:
        return "high"

    if missing_rate < 1.0:
        return "critical"

    return "complete"


class MissingValueDiagnostic(
    AnalysisMethod
):
    name = "missing_value_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose missing values in a column, including "
        "missing rate, missing runs, longest missing run, "
        "and deterministic severity classification."
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
                "Missing-value diagnostics require "
                "at least one observation."
            )

        missing_mask = np.asarray(
            pd.isna(
                series
            ),
            dtype=bool,
        )

        missing_count = int(
            np.sum(
                missing_mask
            )
        )

        complete_count = int(
            total_count
            - missing_count
        )

        missing_rate = float(
            missing_count
            / total_count
        )

        missing_run_count = (
            _count_true_runs(
                missing_mask
            )
        )

        longest_missing_run = (
            _longest_true_run(
                missing_mask
            )
        )

        if missing_count > 0:
            missing_positions = np.flatnonzero(
                missing_mask
            )

            first_missing_index = int(
                missing_positions[
                    0
                ]
            )

            last_missing_index = int(
                missing_positions[
                    -1
                ]
            )
        else:
            first_missing_index = None
            last_missing_index = None

        return {
            "diagnostic_type": (
                "missing_value_diagnostic"
            ),
            "total_count": (
                total_count
            ),
            "complete_count": (
                complete_count
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
            "has_missing": bool(
                missing_count
                > 0
            ),
            "all_missing": bool(
                missing_count
                == total_count
            ),
            "complete": bool(
                missing_count
                == 0
            ),
            "missing_run_count": (
                missing_run_count
            ),
            "longest_missing_run": (
                longest_missing_run
            ),
            "first_missing_index": (
                first_missing_index
            ),
            "last_missing_index": (
                last_missing_index
            ),
            "severity": (
                _missingness_severity(
                    missing_rate
                )
            ),
            "severity_thresholds": {
                "low": "<5%",
                "moderate": "5%-<20%",
                "high": "20%-<50%",
                "critical": "50%-<100%",
                "complete": "100%",
            },
        }
