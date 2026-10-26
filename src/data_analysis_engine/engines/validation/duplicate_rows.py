from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _duplicate_risk_level(
    excess_duplicate_rate: float,
) -> str:
    if excess_duplicate_rate == 0.0:
        return "none"

    if excess_duplicate_rate < 0.01:
        return "low"

    if excess_duplicate_rate < 0.05:
        return "moderate"

    if excess_duplicate_rate < 0.20:
        return "high"

    return "critical"


class DuplicateRowDiagnostic(
    AnalysisMethod
):
    name = "duplicate_row_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Detect exact duplicate rows across compatible "
        "analytical columns while preserving missing "
        "values during duplicate comparison."
    )

    supported_roles = (
        "time",
        "measure",
        "dimension",
        "text",
    )

    minimum_compatible_columns = 1

    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = ()

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = kwargs.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):
            raise InvalidAnalysisInputError(
                "'data' must be a dictionary "
                "of column names to values."
            )

        if not data:
            raise InvalidAnalysisInputError(
                "Duplicate-row diagnostics require "
                "at least one column."
            )

        try:
            frame = pd.DataFrame(
                data
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "All columns must be convertible "
                "to a rectangular dataframe."
            ) from exc

        total_count = int(
            len(
                frame
            )
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Duplicate-row diagnostics require "
                "at least one row."
            )

        comparison_columns = [
            str(
                column
            )
            for column
            in frame.columns
        ]

        duplicate_mask = np.asarray(
            frame.duplicated(
                keep=False
            ),
            dtype=bool,
        )

        excess_mask = np.asarray(
            frame.duplicated(
                keep="first"
            ),
            dtype=bool,
        )

        duplicate_row_count = int(
            np.sum(
                duplicate_mask
            )
        )

        excess_duplicate_count = int(
            np.sum(
                excess_mask
            )
        )

        unique_row_count = int(
            len(
                frame.drop_duplicates()
            )
        )

        if duplicate_row_count > 0:
            duplicate_group_count = int(
                len(
                    frame.loc[
                        duplicate_mask
                    ].drop_duplicates()
                )
            )
        else:
            duplicate_group_count = 0

        duplicate_row_rate = float(
            duplicate_row_count
            / total_count
        )

        excess_duplicate_rate = float(
            excess_duplicate_count
            / total_count
        )

        duplicate_positions = [
            int(
                position
            )
            for position
            in np.flatnonzero(
                duplicate_mask
            )
        ]

        excess_duplicate_positions = [
            int(
                position
            )
            for position
            in np.flatnonzero(
                excess_mask
            )
        ]

        missing_cell_count = int(
            frame.isna().sum().sum()
        )

        total_cell_count = int(
            frame.shape[
                0
            ]
            * frame.shape[
                1
            ]
        )

        missing_cell_rate = float(
            missing_cell_count
            / total_cell_count
        )

        has_duplicates = bool(
            excess_duplicate_count
            > 0
        )

        all_rows_identical = bool(
            total_count > 1
            and unique_row_count == 1
        )

        risk_level = (
            _duplicate_risk_level(
                excess_duplicate_rate
            )
        )

        if not has_duplicates:
            recommendation = (
                "No exact duplicate analytical rows "
                "were detected."
            )

        elif all_rows_identical:
            recommendation = (
                "All rows are identical across the "
                "analytical columns. Review the dataset "
                "before further analysis."
            )

        else:
            recommendation = (
                "Exact duplicate rows were detected. "
                "Review whether they represent legitimate "
                "repeated observations or redundant records "
                "before removing them."
            )

        return {
            "diagnostic_type": (
                "duplicate_row_diagnostic"
            ),
            "comparison_columns": (
                comparison_columns
            ),
            "column_count": int(
                frame.shape[
                    1
                ]
            ),
            "total_count": (
                total_count
            ),
            "unique_row_count": (
                unique_row_count
            ),
            "duplicate_row_count": (
                duplicate_row_count
            ),
            "excess_duplicate_count": (
                excess_duplicate_count
            ),
            "duplicate_group_count": (
                duplicate_group_count
            ),
            "duplicate_row_rate": (
                duplicate_row_rate
            ),
            "excess_duplicate_rate": (
                excess_duplicate_rate
            ),
            "duplicate_positions": (
                duplicate_positions
            ),
            "excess_duplicate_positions": (
                excess_duplicate_positions
            ),
            "has_duplicates": (
                has_duplicates
            ),
            "all_rows_identical": (
                all_rows_identical
            ),
            "missing_cell_count": (
                missing_cell_count
            ),
            "missing_cell_rate": (
                missing_cell_rate
            ),
            "risk_level": (
                risk_level
            ),
            "risk_basis": (
                "excess_duplicate_rate"
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
