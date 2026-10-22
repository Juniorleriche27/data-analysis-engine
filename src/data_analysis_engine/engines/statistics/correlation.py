from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


def _validate_pair(
    x: Any,
    y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        raise InvalidAnalysisInputError(
            "'x' and 'y' are required."
        )

    if not isinstance(x, (list, tuple)):
        raise InvalidAnalysisInputError(
            "'x' must be a list or tuple."
        )

    if not isinstance(y, (list, tuple)):
        raise InvalidAnalysisInputError(
            "'y' must be a list or tuple."
        )

    if len(x) != len(y):
        raise InvalidAnalysisInputError(
            "'x' and 'y' must have the same length."
        )

    if len(x) < 3:
        raise InvalidAnalysisInputError(
            "At least 3 paired observations are required."
        )

    def validate(
        values: list | tuple,
        name: str,
    ) -> np.ndarray:
        validated: list[float] = []

        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise InvalidAnalysisInputError(
                    f"'{name}' must contain only numerical values."
                )

            value = float(value)

            if not isfinite(value):
                raise InvalidAnalysisInputError(
                    f"'{name}' must contain only finite values."
                )

            validated.append(value)

        array = np.asarray(validated, dtype=float)

        if np.all(array == array[0]):
            raise InvalidAnalysisInputError(
                f"'{name}' must not be constant."
            )

        return array

    return validate(x, "x"), validate(y, "y")


def _pearson_coefficient(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()

    numerator = np.sum(
        x_centered * y_centered
    )

    denominator = np.sqrt(
        np.sum(x_centered ** 2)
        * np.sum(y_centered ** 2)
    )

    if denominator == 0:
        raise InvalidAnalysisInputError(
            "Pearson correlation is undefined "
            "for constant input."
        )

    return float(numerator / denominator)


class PearsonCorrelation(AnalysisMethod):
    name = "pearson_correlation"
    category = "statistics"
    version = "1.0"

    description = (
        "Measure the strength and direction of a linear "
        "relationship between two numerical measures."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    allow_constant_columns = False

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_pair(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        coefficient = _pearson_coefficient(
            x,
            y,
        )

        return {
            "coefficient": coefficient,
            "sample_size": len(x),
            "inference": None,
        }


class SpearmanCorrelation(AnalysisMethod):
    name = "spearman_correlation"
    category = "statistics"
    version = "1.0"

    description = (
        "Measure the strength and direction of a monotonic "
        "relationship between two numerical measures using ranks."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    allow_constant_columns = False

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_pair(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        x_rank = (
            pd.Series(x)
            .rank(method="average")
            .to_numpy(dtype=float)
        )

        y_rank = (
            pd.Series(y)
            .rank(method="average")
            .to_numpy(dtype=float)
        )

        coefficient = _pearson_coefficient(
            x_rank,
            y_rank,
        )

        return {
            "coefficient": coefficient,
            "sample_size": len(x),
            "inference": None,
        }
