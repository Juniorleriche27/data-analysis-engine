from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_ALLOWED_STRATEGIES = {
    "log",
    "log1p",
    "sqrt",
    "signed_log1p",
}


def _validate_strategy(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'strategy' must be a string."
        )

    strategy = value.strip().lower()

    if strategy not in _ALLOWED_STRATEGIES:
        raise InvalidAnalysisInputError(
            "'strategy' must be one of: "
            "log, log1p, sqrt, signed_log1p."
        )

    return strategy


def _numeric_values(
    observed: pd.Series,
) -> np.ndarray:
    for value in observed:
        if (
            isinstance(
                value,
                (bool, np.bool_),
            )
            or not isinstance(
                value,
                Real,
            )
        ):
            raise InvalidAnalysisInputError(
                "Numerical transformations require "
                "genuinely numeric non-missing values."
            )

    numeric = np.asarray(
        observed,
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            numeric
        )
    ):
        raise InvalidAnalysisInputError(
            "Numerical transformations require finite "
            "non-missing values. Run "
            "'non_finite_value_diagnostic' first."
        )

    return numeric


def _serialize(
    transformed: np.ndarray,
    missing_mask: np.ndarray,
) -> list[float | None]:
    return [
        (
            None
            if missing_mask[index]
            else float(value)
        )
        for index, value
        in enumerate(
            transformed
        )
    ]


class NumericalTransformation(
    AnalysisMethod
):
    name = "numerical_transformation"
    category = "preprocessing"
    version = "1.0"

    description = (
        "Apply deterministic logarithmic and square-root "
        "transformations to numerical columns while "
        "preserving missing-value positions."
    )

    supported_roles = (
        "measure",
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

        strategy = _validate_strategy(
            kwargs.get(
                "strategy",
                "log1p",
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
            len(series)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Numerical transformations require "
                "at least one observation."
            )

        missing_mask_series = (
            series.isna()
        )

        missing_mask = np.asarray(
            missing_mask_series,
            dtype=bool,
        )

        missing_count = int(
            np.sum(
                missing_mask
            )
        )

        observed = series[
            ~missing_mask_series
        ]

        non_missing_count = int(
            len(
                observed
            )
        )

        if non_missing_count == 0:
            raise InvalidAnalysisInputError(
                "Numerical transformations require "
                "at least one non-missing observation."
            )

        numeric = _numeric_values(
            observed
        )

        if strategy == "log":
            if np.any(
                numeric <= 0.0
            ):
                raise InvalidAnalysisInputError(
                    "Log transformation requires "
                    "all non-missing values to be > 0."
                )

            transformed_values = np.log(
                numeric
            )

            domain_requirement = "x > 0"

        elif strategy == "log1p":
            if np.any(
                numeric <= -1.0
            ):
                raise InvalidAnalysisInputError(
                    "log1p transformation requires "
                    "all non-missing values to be > -1."
                )

            transformed_values = np.log1p(
                numeric
            )

            domain_requirement = "x > -1"

        elif strategy == "sqrt":
            if np.any(
                numeric < 0.0
            ):
                raise InvalidAnalysisInputError(
                    "Square-root transformation requires "
                    "all non-missing values to be >= 0."
                )

            transformed_values = np.sqrt(
                numeric
            )

            domain_requirement = "x >= 0"

        else:
            transformed_values = (
                np.sign(
                    numeric
                )
                * np.log1p(
                    np.abs(
                        numeric
                    )
                )
            )

            domain_requirement = (
                "all finite real values"
            )

        if not np.all(
            np.isfinite(
                transformed_values
            )
        ):
            raise InvalidAnalysisInputError(
                "Transformation produced non-finite values."
            )

        transformed = np.full(
            total_count,
            np.nan,
            dtype=float,
        )

        observed_positions = np.flatnonzero(
            ~missing_mask
        )

        transformed[
            observed_positions
        ] = transformed_values

        return {
            "transformation_type": (
                "numerical_transformation"
            ),
            "strategy": strategy,
            "domain_requirement": (
                domain_requirement
            ),
            "total_count": (
                total_count
            ),
            "non_missing_count": (
                non_missing_count
            ),
            "missing_count": (
                missing_count
            ),
            "missing_positions": [
                int(position)
                for position
                in np.flatnonzero(
                    missing_mask
                )
            ],
            "input_minimum": float(
                np.min(
                    numeric
                )
            ),
            "input_maximum": float(
                np.max(
                    numeric
                )
            ),
            "output_minimum": float(
                np.min(
                    transformed_values
                )
            ),
            "output_maximum": float(
                np.max(
                    transformed_values
                )
            ),
            "values": _serialize(
                transformed,
                missing_mask,
            ),
        }
