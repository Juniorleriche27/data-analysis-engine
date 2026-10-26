from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_quantile(
    name: str,
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be numeric."
        )

    quantile = float(value)

    if (
        not np.isfinite(quantile)
        or quantile < 0.0
        or quantile > 1.0
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must satisfy 0 <= {name} <= 1."
        )

    return quantile


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
                "Winsorization requires genuinely numeric "
                "non-missing values."
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
            "Winsorization requires finite non-missing "
            "values. Run 'non_finite_value_diagnostic' first."
        )

    return numeric


class Winsorization(
    AnalysisMethod
):
    name = "winsorization"
    category = "preprocessing"
    version = "1.0"

    description = (
        "Clip numerical observations to configurable lower "
        "and upper empirical quantile bounds while preserving "
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

        lower_quantile = _validate_quantile(
            "lower_quantile",
            kwargs.get(
                "lower_quantile",
                0.01,
            ),
        )

        upper_quantile = _validate_quantile(
            "upper_quantile",
            kwargs.get(
                "upper_quantile",
                0.99,
            ),
        )

        if lower_quantile >= upper_quantile:
            raise InvalidAnalysisInputError(
                "'lower_quantile' must be strictly less "
                "than 'upper_quantile'."
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
                "Winsorization requires at least "
                "one observation."
            )

        missing_mask_series = series.isna()

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
            len(observed)
        )

        if non_missing_count == 0:
            raise InvalidAnalysisInputError(
                "Winsorization requires at least one "
                "non-missing observation."
            )

        numeric = _numeric_values(
            observed
        )

        lower_bound = float(
            np.quantile(
                numeric,
                lower_quantile,
            )
        )

        upper_bound = float(
            np.quantile(
                numeric,
                upper_quantile,
            )
        )

        winsorized_values = np.clip(
            numeric,
            lower_bound,
            upper_bound,
        )

        lower_changed_mask = (
            numeric
            < lower_bound
        )

        upper_changed_mask = (
            numeric
            > upper_bound
        )

        changed_mask = (
            lower_changed_mask
            | upper_changed_mask
        )

        observed_positions = np.flatnonzero(
            ~missing_mask
        )

        lower_clipped_positions = [
            int(position)
            for position
            in observed_positions[
                lower_changed_mask
            ]
        ]

        upper_clipped_positions = [
            int(position)
            for position
            in observed_positions[
                upper_changed_mask
            ]
        ]

        clipped_positions = [
            int(position)
            for position
            in observed_positions[
                changed_mask
            ]
        ]

        transformed = np.full(
            total_count,
            np.nan,
            dtype=float,
        )

        transformed[
            observed_positions
        ] = winsorized_values

        serialized_values: list[
            float | None
        ] = []

        for index, value in enumerate(
            transformed
        ):
            if missing_mask[index]:
                serialized_values.append(
                    None
                )
            else:
                serialized_values.append(
                    float(value)
                )

        lower_clipped_count = int(
            np.sum(
                lower_changed_mask
            )
        )

        upper_clipped_count = int(
            np.sum(
                upper_changed_mask
            )
        )

        clipped_count = int(
            np.sum(
                changed_mask
            )
        )

        return {
            "transformation_type": (
                "winsorization"
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
            "lower_quantile": (
                lower_quantile
            ),
            "upper_quantile": (
                upper_quantile
            ),
            "lower_bound": (
                lower_bound
            ),
            "upper_bound": (
                upper_bound
            ),
            "lower_clipped_count": (
                lower_clipped_count
            ),
            "upper_clipped_count": (
                upper_clipped_count
            ),
            "clipped_count": (
                clipped_count
            ),
            "lower_clipped_positions": (
                lower_clipped_positions
            ),
            "upper_clipped_positions": (
                upper_clipped_positions
            ),
            "clipped_positions": (
                clipped_positions
            ),
            "changed": bool(
                clipped_count > 0
            ),
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
                    winsorized_values
                )
            ),
            "output_maximum": float(
                np.max(
                    winsorized_values
                )
            ),
            "values": (
                serialized_values
            ),
        }
