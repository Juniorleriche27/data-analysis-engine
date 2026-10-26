from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_ALLOWED_STRATEGIES = {
    "mean",
    "median",
    "mode",
    "constant",
}

_MISSING = object()


def _to_python_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


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

    strategy = (
        value.strip().lower()
    )

    if strategy not in _ALLOWED_STRATEGIES:
        raise InvalidAnalysisInputError(
            "'strategy' must be one of: "
            "mean, median, mode, constant."
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
                "Mean and median imputation require "
                "genuinely numeric non-missing values."
            )

    values = np.asarray(
        observed,
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise InvalidAnalysisInputError(
            "Imputation requires finite non-missing "
            "numeric values."
        )

    return values


def _mode_value(
    observed: pd.Series,
) -> Any:
    try:
        counts = observed.value_counts(
            sort=False,
            dropna=False,
        )
    except TypeError as exc:
        raise InvalidAnalysisInputError(
            "Mode imputation requires hashable "
            "scalar observations."
        ) from exc

    maximum_count = int(
        counts.max()
    )

    for value in observed:
        if int(
            counts.loc[value]
        ) == maximum_count:
            return value

    raise InvalidAnalysisInputError(
        "Unable to determine a mode."
    )


class MissingValueImputation(
    AnalysisMethod
):
    name = "missing_value_imputation"
    category = "preprocessing"
    version = "1.0"

    description = (
        "Impute missing values using deterministic mean, "
        "median, mode, or constant strategies while "
        "reporting exactly which observations changed."
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

        strategy = _validate_strategy(
            kwargs.get(
                "strategy",
                "median",
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
                "Imputation requires at least "
                "one observation."
            )

        missing_mask = series.isna()

        missing_count = int(
            missing_mask.sum()
        )

        observed = series[
            ~missing_mask
        ]

        non_missing_count = int(
            len(observed)
        )

        if missing_count == 0:
            return {
                "transformation_type": (
                    "missing_value_imputation"
                ),
                "strategy": strategy,
                "total_count": total_count,
                "non_missing_count_before": (
                    non_missing_count
                ),
                "missing_count_before": 0,
                "missing_count_after": 0,
                "imputed_count": 0,
                "imputed_positions": [],
                "fill_value": None,
                "changed": False,
                "values": [
                    _to_python_scalar(
                        value
                    )
                    for value
                    in series
                ],
            }

        if (
            non_missing_count == 0
            and strategy != "constant"
        ):
            raise InvalidAnalysisInputError(
                "Mean, median, and mode imputation "
                "require at least one non-missing value."
            )

        if strategy == "mean":
            numeric = _numeric_values(
                observed
            )

            fill_value = float(
                np.mean(
                    numeric
                )
            )

        elif strategy == "median":
            numeric = _numeric_values(
                observed
            )

            fill_value = float(
                np.median(
                    numeric
                )
            )

        elif strategy == "mode":
            fill_value = _mode_value(
                observed
            )

        else:
            fill_value = kwargs.get(
                "fill_value",
                _MISSING,
            )

            if fill_value is _MISSING:
                raise InvalidAnalysisInputError(
                    "'fill_value' is required when "
                    "strategy='constant'."
                )

            try:
                missing_fill = bool(
                    pd.isna(
                        fill_value
                    )
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise InvalidAnalysisInputError(
                    "'fill_value' must be a scalar value."
                ) from exc

            if missing_fill:
                raise InvalidAnalysisInputError(
                    "'fill_value' cannot itself be missing."
                )

        imputed_positions = [
            int(position)
            for position
            in np.flatnonzero(
                np.asarray(
                    missing_mask,
                    dtype=bool,
                )
            )
        ]

        transformed = series.copy()

        transformed.loc[
            missing_mask
        ] = fill_value

        missing_count_after = int(
            transformed.isna().sum()
        )

        return {
            "transformation_type": (
                "missing_value_imputation"
            ),
            "strategy": strategy,
            "total_count": total_count,
            "non_missing_count_before": (
                non_missing_count
            ),
            "missing_count_before": (
                missing_count
            ),
            "missing_count_after": (
                missing_count_after
            ),
            "imputed_count": (
                missing_count
                - missing_count_after
            ),
            "imputed_positions": (
                imputed_positions
            ),
            "fill_value": (
                _to_python_scalar(
                    fill_value
                )
            ),
            "changed": bool(
                missing_count_after
                < missing_count
            ),
            "values": [
                _to_python_scalar(
                    value
                )
                for value
                in transformed
            ],
        }
