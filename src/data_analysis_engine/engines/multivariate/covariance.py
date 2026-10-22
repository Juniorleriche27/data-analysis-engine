from math import isfinite, sqrt
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _validate_multivariate_data(
    data: Any,
) -> tuple[list[str], np.ndarray]:
    """
    Validate a mapping of variable names to numerical observations.

    Expected format:
    {
        "x": [1, 2, 3],
        "y": [4, 5, 6],
    }
    """

    if not isinstance(data, dict):
        raise InvalidAnalysisInputError(
            "'data' must be a dictionary mapping "
            "column names to numerical observations."
        )

    if len(data) < 2:
        raise InvalidAnalysisInputError(
            "At least two variables are required."
        )

    columns = list(data.keys())

    if not all(
        isinstance(column, str) and column.strip()
        for column in columns
    ):
        raise InvalidAnalysisInputError(
            "All variable names must be non-empty strings."
        )

    lengths: set[int] = set()
    validated_columns: list[list[float]] = []

    for column in columns:
        values = data[column]

        if not isinstance(values, (list, tuple, np.ndarray)):
            raise InvalidAnalysisInputError(
                f"'{column}' must contain a numerical sequence."
            )

        if len(values) < 2:
            raise InvalidAnalysisInputError(
                f"'{column}' must contain at least 2 observations."
            )

        validated: list[float] = []

        for value in values:
            if (
                not isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                or isinstance(value, (bool, np.bool_))
            ):
                raise InvalidAnalysisInputError(
                    f"'{column}' must contain only numerical values."
                )

            numeric_value = float(value)

            if not isfinite(numeric_value):
                raise InvalidAnalysisInputError(
                    f"'{column}' must contain only finite values."
                )

            validated.append(numeric_value)

        array = np.asarray(validated, dtype=float)

        if np.all(array == array[0]):
            raise InvalidAnalysisInputError(
                f"'{column}' must not be constant."
            )

        lengths.add(len(validated))
        validated_columns.append(validated)

    if len(lengths) != 1:
        raise InvalidAnalysisInputError(
            "All variables must contain the same number of observations."
        )

    matrix = np.column_stack(validated_columns)

    return columns, matrix


def _sample_covariance_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute the sample covariance matrix using ddof=1.
    """

    covariance = np.cov(
        matrix,
        rowvar=False,
        ddof=1,
    )

    return np.asarray(
        covariance,
        dtype=float,
    )


class CovarianceMatrix(AnalysisMethod):
    name = "covariance_matrix"
    category = "multivariate"
    version = "1.0"

    description = (
        "Compute the sample covariance matrix of two or more "
        "numerical measures."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "matrix"
    allow_constant_columns = False

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        columns, matrix = _validate_multivariate_data(
            kwargs.get("data")
        )

        covariance = _sample_covariance_matrix(
            matrix
        )

        return {
            "columns": columns,
            "sample_size": int(matrix.shape[0]),
            "variable_count": int(matrix.shape[1]),
            "ddof": 1,
            "covariance_matrix": covariance.tolist(),
        }


class LinearCombinationVariance(AnalysisMethod):
    name = "linear_combination_variance"
    category = "multivariate"
    version = "1.0"

    description = (
        "Compute the variance of a linear combination "
        "using w^T Sigma w and decompose each variable's "
        "contribution to total variance."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "matrix"
    allow_constant_columns = False

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        columns, matrix = _validate_multivariate_data(
            kwargs.get("data")
        )

        weights = kwargs.get("weights")

        if not isinstance(weights, dict):
            raise InvalidAnalysisInputError(
                "'weights' must be a dictionary mapping "
                "variable names to numerical weights."
            )

        if set(weights.keys()) != set(columns):
            raise InvalidAnalysisInputError(
                "'weights' must contain exactly the same "
                "variable names as 'data'."
            )

        weight_values: list[float] = []

        for column in columns:
            value = weights[column]

            if (
                not isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                or isinstance(value, (bool, np.bool_))
            ):
                raise InvalidAnalysisInputError(
                    f"Weight for '{column}' must be numerical."
                )

            numeric_weight = float(value)

            if not isfinite(numeric_weight):
                raise InvalidAnalysisInputError(
                    f"Weight for '{column}' must be finite."
                )

            weight_values.append(numeric_weight)

        weight_vector = np.asarray(
            weight_values,
            dtype=float,
        )

        covariance = _sample_covariance_matrix(
            matrix
        )

        covariance_times_weights = (
            covariance @ weight_vector
        )

        variance = float(
            weight_vector
            @ covariance_times_weights
        )

        if variance < 0 and abs(variance) < 1e-12:
            variance = 0.0

        if variance < 0:
            raise InvalidAnalysisInputError(
                "Computed variance is negative. "
                "The covariance matrix may be numerically invalid."
            )

        standard_deviation = sqrt(variance)

        component_contributions = (
            weight_vector
            * covariance_times_weights
        )

        contributions: dict[str, dict[str, float | None]] = {}

        for index, column in enumerate(columns):
            contribution = float(
                component_contributions[index]
            )

            percentage = (
                contribution / variance
                if variance > 0
                else None
            )

            contributions[column] = {
                "weight": float(weight_vector[index]),
                "marginal_variance_effect": float(
                    covariance_times_weights[index]
                ),
                "variance_contribution": contribution,
                "contribution_percentage": percentage,
            }

        return {
            "columns": columns,
            "weights": {
                column: float(weights[column])
                for column in columns
            },
            "sample_size": int(matrix.shape[0]),
            "ddof": 1,
            "covariance_matrix": covariance.tolist(),
            "variance": variance,
            "standard_deviation": standard_deviation,
            "contributions": contributions,
        }
