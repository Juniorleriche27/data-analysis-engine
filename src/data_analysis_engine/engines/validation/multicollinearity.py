from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _compute_vif(
    matrix: np.ndarray,
    column_index: int,
) -> float:
    target = matrix[
        :,
        column_index,
    ]

    predictors = np.delete(
        matrix,
        column_index,
        axis=1,
    )

    design = np.column_stack(
        [
            np.ones(
                matrix.shape[0],
                dtype=float,
            ),
            predictors,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        target,
        rcond=None,
    )

    fitted = (
        design
        @ coefficients
    )

    residual = (
        target
        - fitted
    )

    residual_sum_squares = float(
        np.sum(
            residual ** 2
        )
    )

    centered = (
        target
        - np.mean(target)
    )

    total_sum_squares = float(
        np.sum(
            centered ** 2
        )
    )

    if total_sum_squares == 0.0:
        return float("inf")

    r_squared = float(
        1.0
        - residual_sum_squares
        / total_sum_squares
    )

    r_squared = min(
        max(
            r_squared,
            0.0,
        ),
        1.0,
    )

    denominator = (
        1.0
        - r_squared
    )

    if denominator <= 1e-12:
        return float("inf")

    return float(
        1.0
        / denominator
    )


def _risk_level(
    maximum_vif: float,
    condition_number: float,
    maximum_absolute_correlation: float,
) -> str:
    if (
        np.isinf(maximum_vif)
        or np.isinf(condition_number)
        or maximum_vif >= 10.0
        or condition_number >= 30.0
        or maximum_absolute_correlation >= 0.90
    ):
        return "critical"

    if (
        maximum_vif >= 5.0
        or condition_number >= 10.0
        or maximum_absolute_correlation >= 0.80
    ):
        return "high"

    return "none"


class MulticollinearityDiagnostic(
    AnalysisMethod
):
    name = "multicollinearity_diagnostic"
    category = "validation"
    version = "1.0"

    description = (
        "Diagnose multicollinearity across numerical "
        "analytical columns using pairwise correlations, "
        "condition number, and variance inflation factors."
    )

    supported_roles = (
        "measure",
    )

    minimum_compatible_columns = 2

    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = (
        (
            "measure",
            2,
        ),
    )

    allow_constant_columns = False
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = kwargs.get("data")

        if not isinstance(
            data,
            dict,
        ):
            raise InvalidAnalysisInputError(
                "'data' must be a dictionary "
                "of column names to values."
            )

        if len(data) < 2:
            raise InvalidAnalysisInputError(
                "Multicollinearity diagnostics require "
                "at least two numerical columns."
            )

        try:
            frame = pd.DataFrame(
                data
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "All columns must form a rectangular "
                "dataframe."
            ) from exc

        total_count = int(
            len(frame)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Multicollinearity diagnostics require "
                "at least one row."
            )

        columns = [
            str(column)
            for column
            in frame.columns
        ]

        try:
            numeric = frame.astype(
                float
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise InvalidAnalysisInputError(
                "Multicollinearity diagnostics require "
                "numeric columns."
            ) from exc

        finite_or_missing = (
            np.isfinite(
                numeric.to_numpy(
                    dtype=float
                )
            )
            | numeric.isna().to_numpy()
        )

        if not np.all(
            finite_or_missing
        ):
            raise InvalidAnalysisInputError(
                "Multicollinearity diagnostics require "
                "finite non-missing values. Run "
                "'non_finite_value_diagnostic' first."
            )

        complete = numeric.dropna(
            axis=0,
            how="any",
        )

        complete_row_count = int(
            len(complete)
        )

        dropped_row_count = int(
            total_count
            - complete_row_count
        )

        complete_case_rate = float(
            complete_row_count
            / total_count
        )

        column_count = int(
            complete.shape[1]
        )

        if (
            complete_row_count
            < column_count + 1
        ):
            raise InvalidAnalysisInputError(
                "Multicollinearity diagnostics require "
                "at least column_count + 1 complete rows."
            )

        matrix = complete.to_numpy(
            dtype=float
        )

        means = np.mean(
            matrix,
            axis=0,
        )

        standard_deviations = np.std(
            matrix,
            axis=0,
            ddof=0,
        )

        zero_variance_columns = [
            columns[index]
            for index, standard_deviation
            in enumerate(
                standard_deviations
            )
            if standard_deviation == 0.0
        ]

        if zero_variance_columns:
            raise InvalidAnalysisInputError(
                "Multicollinearity diagnostics cannot "
                "include zero-variance columns: "
                + ", ".join(
                    zero_variance_columns
                )
                + ". Run 'variance_diagnostic' first."
            )

        standardized = (
            matrix
            - means
        ) / standard_deviations

        correlation_matrix = np.corrcoef(
            standardized,
            rowvar=False,
        )

        off_diagonal_mask = ~np.eye(
            column_count,
            dtype=bool,
        )

        absolute_off_diagonal = np.abs(
            correlation_matrix[
                off_diagonal_mask
            ]
        )

        maximum_absolute_correlation = float(
            np.max(
                absolute_off_diagonal
            )
        )

        maximum_pair = None
        maximum_pair_value = -1.0

        for left in range(
            column_count
        ):
            for right in range(
                left + 1,
                column_count,
            ):
                value = abs(
                    float(
                        correlation_matrix[
                            left,
                            right,
                        ]
                    )
                )

                if value > maximum_pair_value:
                    maximum_pair_value = value
                    maximum_pair = [
                        columns[left],
                        columns[right],
                    ]

        singular_values = np.linalg.svd(
            standardized,
            compute_uv=False,
        )

        maximum_singular_value = float(
            np.max(
                singular_values
            )
        )

        minimum_singular_value = float(
            np.min(
                singular_values
            )
        )

        if (
            minimum_singular_value
            <= 1e-12
        ):
            condition_number = float(
                "inf"
            )
        else:
            condition_number = float(
                maximum_singular_value
                / minimum_singular_value
            )

        vif_by_column = {}

        for index, column in enumerate(
            columns
        ):
            vif_by_column[column] = (
                _compute_vif(
                    matrix,
                    index,
                )
            )

        maximum_vif = max(
            vif_by_column.values()
        )

        problematic_vif_columns = [
            column
            for column, value
            in vif_by_column.items()
            if (
                np.isinf(value)
                or value >= 5.0
            )
        ]

        highly_correlated_pairs = []

        for left in range(
            column_count
        ):
            for right in range(
                left + 1,
                column_count,
            ):
                correlation = float(
                    correlation_matrix[
                        left,
                        right,
                    ]
                )

                if abs(
                    correlation
                ) >= 0.80:
                    highly_correlated_pairs.append(
                        {
                            "columns": [
                                columns[left],
                                columns[right],
                            ],
                            "correlation": (
                                correlation
                            ),
                        }
                    )

        risk_level = _risk_level(
            maximum_vif,
            condition_number,
            maximum_absolute_correlation,
        )

        has_multicollinearity = bool(
            risk_level
            in {
                "high",
                "critical",
            }
        )

        if risk_level == "critical":
            recommendation = (
                "Severe multicollinearity was detected. "
                "Review redundant predictors before "
                "regression or other coefficient-sensitive "
                "models."
            )

        elif risk_level == "high":
            recommendation = (
                "Material multicollinearity was detected. "
                "Review correlated predictors and VIF "
                "before modelling."
            )

        else:
            recommendation = (
                "No material multicollinearity problem "
                "was detected."
            )

        correlation_by_column = {
            columns[row]: {
                columns[column]: float(
                    correlation_matrix[
                        row,
                        column,
                    ]
                )
                for column
                in range(
                    column_count
                )
            }
            for row
            in range(
                column_count
            )
        }

        serializable_vif = {
            column: (
                None
                if np.isinf(value)
                else float(value)
            )
            for column, value
            in vif_by_column.items()
        }

        infinite_vif_columns = [
            column
            for column, value
            in vif_by_column.items()
            if np.isinf(value)
        ]

        return {
            "diagnostic_type": (
                "multicollinearity_diagnostic"
            ),
            "columns": columns,
            "column_count": column_count,
            "total_count": total_count,
            "complete_row_count": (
                complete_row_count
            ),
            "dropped_row_count": (
                dropped_row_count
            ),
            "complete_case_rate": (
                complete_case_rate
            ),
            "correlation_matrix": (
                correlation_by_column
            ),
            "maximum_absolute_correlation": (
                maximum_absolute_correlation
            ),
            "maximum_correlation_pair": (
                maximum_pair
            ),
            "highly_correlated_pairs": (
                highly_correlated_pairs
            ),
            "condition_number": (
                None
                if np.isinf(
                    condition_number
                )
                else condition_number
            ),
            "condition_number_infinite": bool(
                np.isinf(
                    condition_number
                )
            ),
            "vif_by_column": (
                serializable_vif
            ),
            "infinite_vif_columns": (
                infinite_vif_columns
            ),
            "maximum_vif": (
                None
                if np.isinf(
                    maximum_vif
                )
                else float(
                    maximum_vif
                )
            ),
            "maximum_vif_infinite": bool(
                np.isinf(
                    maximum_vif
                )
            ),
            "problematic_vif_columns": (
                problematic_vif_columns
            ),
            "has_multicollinearity": (
                has_multicollinearity
            ),
            "risk_level": (
                risk_level
            ),
            "recommendation": (
                recommendation
            ),
        }
