from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_ALLOWED_STRATEGIES = {
    "standard",
    "minmax",
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
            "standard, minmax."
        )

    return strategy


def _validate_feature_range(
    value: Any,
) -> tuple[float, float]:
    if (
        not isinstance(
            value,
            (tuple, list),
        )
        or len(value) != 2
    ):
        raise InvalidAnalysisInputError(
            "'feature_range' must contain "
            "exactly two numeric values."
        )

    lower, upper = value

    for item in (
        lower,
        upper,
    ):
        if (
            isinstance(
                item,
                bool,
            )
            or not isinstance(
                item,
                Real,
            )
        ):
            raise InvalidAnalysisInputError(
                "'feature_range' values must "
                "be numeric."
            )

    lower = float(lower)
    upper = float(upper)

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower >= upper
    ):
        raise InvalidAnalysisInputError(
            "'feature_range' must contain finite "
            "values with lower < upper."
        )

    return (
        lower,
        upper,
    )


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
                "Feature scaling requires genuinely "
                "numeric non-missing values."
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
            "Feature scaling requires finite "
            "non-missing values. Run "
            "'non_finite_value_diagnostic' first."
        )

    return numeric


def _serialize_values(
    values: np.ndarray,
    missing_mask: np.ndarray,
) -> list[float | None]:
    result: list[
        float | None
    ] = []

    for index, value in enumerate(
        values
    ):
        if missing_mask[index]:
            result.append(
                None
            )
        else:
            result.append(
                float(value)
            )

    return result


class FeatureScaling(
    AnalysisMethod
):
    name = "feature_scaling"
    category = "preprocessing"
    version = "1.0"

    description = (
        "Scale numerical features using population "
        "z-score standardization or deterministic "
        "min-max transformation while preserving "
        "missing-value positions."
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
                "standard",
            )
        )

        feature_range = (
            _validate_feature_range(
                kwargs.get(
                    "feature_range",
                    (
                        0.0,
                        1.0,
                    ),
                )
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
                "Feature scaling requires "
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
                "Feature scaling requires at least "
                "one non-missing observation."
            )

        numeric = _numeric_values(
            observed
        )

        minimum = float(
            np.min(
                numeric
            )
        )

        maximum = float(
            np.max(
                numeric
            )
        )

        mean = float(
            np.mean(
                numeric
            )
        )

        standard_deviation = float(
            np.std(
                numeric,
                ddof=0,
            )
        )

        is_constant = bool(
            maximum == minimum
        )

        transformed = np.full(
            total_count,
            np.nan,
            dtype=float,
        )

        observed_positions = np.flatnonzero(
            ~missing_mask
        )

        if strategy == "standard":
            if standard_deviation == 0.0:
                transformed_values = np.zeros(
                    non_missing_count,
                    dtype=float,
                )
            else:
                transformed_values = (
                    numeric
                    - mean
                ) / standard_deviation

            parameters = {
                "mean": mean,
                "standard_deviation": (
                    standard_deviation
                ),
            }

        else:
            lower, upper = (
                feature_range
            )

            if maximum == minimum:
                transformed_values = np.full(
                    non_missing_count,
                    lower,
                    dtype=float,
                )
            else:
                transformed_values = (
                    lower
                    + (
                        numeric
                        - minimum
                    )
                    * (
                        upper
                        - lower
                    )
                    / (
                        maximum
                        - minimum
                    )
                )

            parameters = {
                "minimum": minimum,
                "maximum": maximum,
                "feature_range": [
                    lower,
                    upper,
                ],
            }

        transformed[
            observed_positions
        ] = transformed_values

        return {
            "transformation_type": (
                "feature_scaling"
            ),
            "strategy": strategy,
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
            "input_minimum": minimum,
            "input_maximum": maximum,
            "input_mean": mean,
            "input_standard_deviation": (
                standard_deviation
            ),
            "is_constant": (
                is_constant
            ),
            "parameters": (
                parameters
            ),
            "values": _serialize_values(
                transformed,
                missing_mask,
            ),
        }
