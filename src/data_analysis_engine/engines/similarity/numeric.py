from math import isfinite
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_vectors(
    x: Any,
    y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        raise InvalidAnalysisInputError(
            "'x' and 'y' are required."
        )

    if not isinstance(x, (list, tuple, np.ndarray)):
        raise InvalidAnalysisInputError(
            "'x' must be a numerical sequence."
        )

    if not isinstance(y, (list, tuple, np.ndarray)):
        raise InvalidAnalysisInputError(
            "'y' must be a numerical sequence."
        )

    if len(x) != len(y):
        raise InvalidAnalysisInputError(
            "'x' and 'y' must have the same length."
        )

    if len(x) == 0:
        raise InvalidAnalysisInputError(
            "Vectors cannot be empty."
        )

    def validate(
        values: list | tuple | np.ndarray,
        name: str,
    ) -> np.ndarray:
        validated: list[float] = []

        for value in values:
            if (
                not isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
                or isinstance(value, (bool, np.bool_))
            ):
                raise InvalidAnalysisInputError(
                    f"'{name}' must contain only numerical values."
                )

            numeric_value = float(value)

            if not isfinite(numeric_value):
                raise InvalidAnalysisInputError(
                    f"'{name}' must contain only finite values."
                )

            validated.append(numeric_value)

        return np.asarray(
            validated,
            dtype=float,
        )

    return validate(x, "x"), validate(y, "y")


class EuclideanDistance(AnalysisMethod):
    name = "euclidean_distance"
    category = "similarity"
    version = "1.0"

    description = (
        "Compute Euclidean distance between two numerical vectors."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    analysis_unit = "vector"
    allow_constant_columns = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_vectors(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        return {
            "distance": float(
                np.linalg.norm(x - y)
            ),
            "dimension": len(x),
        }


class ManhattanDistance(AnalysisMethod):
    name = "manhattan_distance"
    category = "similarity"
    version = "1.0"

    description = (
        "Compute Manhattan distance between two numerical vectors."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    analysis_unit = "vector"
    allow_constant_columns = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_vectors(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        return {
            "distance": float(
                np.sum(np.abs(x - y))
            ),
            "dimension": len(x),
        }


class MinkowskiDistance(AnalysisMethod):
    name = "minkowski_distance"
    category = "similarity"
    version = "1.0"

    description = (
        "Compute Minkowski distance of order p "
        "between two numerical vectors."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    analysis_unit = "vector"
    allow_constant_columns = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_vectors(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        p = kwargs.get("p", 2.0)

        if (
            not isinstance(
                p,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
            or isinstance(p, (bool, np.bool_))
        ):
            raise InvalidAnalysisInputError(
                "'p' must be numerical."
            )

        p = float(p)

        if not isfinite(p) or p < 1:
            raise InvalidAnalysisInputError(
                "'p' must be finite and greater than or equal to 1."
            )

        return {
            "distance": float(
                np.sum(
                    np.abs(x - y) ** p
                ) ** (1.0 / p)
            ),
            "dimension": len(x),
            "p": p,
        }


def _cosine_similarity(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))

    if x_norm == 0 or y_norm == 0:
        raise InvalidAnalysisInputError(
            "Cosine similarity is undefined for a zero vector."
        )

    similarity = float(
        np.dot(x, y)
        / (x_norm * y_norm)
    )

    return float(
        np.clip(
            similarity,
            -1.0,
            1.0,
        )
    )


class CosineSimilarity(AnalysisMethod):
    name = "cosine_similarity"
    category = "similarity"
    version = "1.0"

    description = (
        "Measure angular similarity between two numerical vectors."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    analysis_unit = "vector"
    allow_constant_columns = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_vectors(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        return {
            "similarity": _cosine_similarity(
                x,
                y,
            ),
            "dimension": len(x),
        }


class CosineDistance(AnalysisMethod):
    name = "cosine_distance"
    category = "similarity"
    version = "1.0"

    description = (
        "Compute cosine distance as one minus cosine similarity."
    )

    supported_roles = ("measure",)
    minimum_compatible_columns = 2
    selection_mode = "pairwise"
    analysis_unit = "vector"
    allow_constant_columns = True

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        x, y = _validate_vectors(
            kwargs.get("x"),
            kwargs.get("y"),
        )

        similarity = _cosine_similarity(
            x,
            y,
        )

        return {
            "distance": float(
                1.0 - similarity
            ),
            "similarity": similarity,
            "dimension": len(x),
        }
