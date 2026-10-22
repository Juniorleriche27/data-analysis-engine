from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.multivariate.covariance import (
    _validate_multivariate_data,
)


class PrincipalComponentAnalysis(AnalysisMethod):
    name = "principal_component_analysis"
    category = "multivariate"
    version = "1.0"

    description = (
        "Perform principal component analysis on two or more "
        "numerical variables using eigen decomposition."
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

        standardize = kwargs.get(
            "standardize",
            True,
        )

        if not isinstance(standardize, bool):
            raise InvalidAnalysisInputError(
                "'standardize' must be a boolean."
            )

        n_samples, n_features = matrix.shape

        max_components = min(
            n_samples - 1,
            n_features,
        )

        if max_components < 1:
            raise InvalidAnalysisInputError(
                "PCA requires at least two observations."
            )

        n_components = kwargs.get(
            "n_components"
        )

        if n_components is None:
            n_components = max_components

        if (
            not isinstance(n_components, int)
            or isinstance(n_components, bool)
        ):
            raise InvalidAnalysisInputError(
                "'n_components' must be an integer."
            )

        if (
            n_components < 1
            or n_components > max_components
        ):
            raise InvalidAnalysisInputError(
                "'n_components' must be between 1 and "
                f"{max_components}."
            )

        means = matrix.mean(
            axis=0
        )

        centered = (
            matrix
            - means
        )

        if standardize:
            scales = matrix.std(
                axis=0,
                ddof=1,
            )

            if np.any(scales == 0):
                raise InvalidAnalysisInputError(
                    "PCA cannot standardize constant variables."
                )

            transformed = (
                centered
                / scales
            )

        else:
            scales = np.ones(
                n_features,
                dtype=float,
            )

            transformed = centered

        covariance_matrix = np.cov(
            transformed,
            rowvar=False,
            ddof=1,
        )

        covariance_matrix = np.asarray(
            covariance_matrix,
            dtype=float,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance_matrix
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[
            order
        ]

        eigenvectors = eigenvectors[
            :,
            order
        ]

        # Numerical protection against tiny negative
        # eigenvalues caused by floating-point arithmetic.
        eigenvalues = np.where(
            (
                eigenvalues < 0
            )
            & (
                np.abs(eigenvalues) < 1e-12
            ),
            0.0,
            eigenvalues,
        )

        if np.any(eigenvalues < 0):
            raise InvalidAnalysisInputError(
                "PCA produced materially negative eigenvalues."
            )

        total_variance = float(
            eigenvalues.sum()
        )

        if total_variance <= 0:
            raise InvalidAnalysisInputError(
                "Total variance must be positive for PCA."
            )

        explained_variance_ratio = (
            eigenvalues
            / total_variance
        )

        cumulative_ratio = np.cumsum(
            explained_variance_ratio
        )

        selected_vectors = eigenvectors[
            :,
            :n_components
        ]

        scores = (
            transformed
            @ selected_vectors
        )

        components: list[dict[str, Any]] = []

        for index in range(
            n_components
        ):
            components.append(
                {
                    "component": (
                        f"PC{index + 1}"
                    ),
                    "eigenvalue": float(
                        eigenvalues[index]
                    ),
                    "explained_variance_ratio": float(
                        explained_variance_ratio[index]
                    ),
                    "cumulative_explained_variance_ratio": float(
                        cumulative_ratio[index]
                    ),
                    "loadings": {
                        column: float(
                            selected_vectors[
                                column_index,
                                index,
                            ]
                        )
                        for (
                            column_index,
                            column
                        ) in enumerate(columns)
                    },
                }
            )

        score_records = [
            {
                f"PC{index + 1}": float(
                    scores[
                        row_index,
                        index,
                    ]
                )
                for index in range(
                    n_components
                )
            }
            for row_index in range(
                n_samples
            )
        ]

        return {
            "columns": columns,
            "sample_size": n_samples,
            "variable_count": n_features,
            "n_components": n_components,
            "standardized": standardize,
            "means": {
                column: float(
                    means[index]
                )
                for index, column in enumerate(
                    columns
                )
            },
            "scales": {
                column: float(
                    scales[index]
                )
                for index, column in enumerate(
                    columns
                )
            },
            "covariance_matrix": (
                covariance_matrix.tolist()
            ),
            "eigenvalues": [
                float(value)
                for value in eigenvalues
            ],
            "explained_variance_ratio": [
                float(value)
                for value in explained_variance_ratio
            ],
            "components": components,
            "scores": score_records,
        }
