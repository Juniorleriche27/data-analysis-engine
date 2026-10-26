from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_probabilities(
    value: Any,
) -> list[float]:
    if value is None:
        return [
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]

    if not isinstance(
        value,
        (list, tuple),
    ):
        raise InvalidAnalysisInputError(
            "'probabilities' must be a list or tuple."
        )

    if not value:
        raise InvalidAnalysisInputError(
            "'probabilities' cannot be empty."
        )

    probabilities: list[float] = []

    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, Real)
        ):
            raise InvalidAnalysisInputError(
                "Every quantile probability must be numeric."
            )

        probability = float(item)

        if (
            not np.isfinite(probability)
            or probability < 0.0
            or probability > 1.0
        ):
            raise InvalidAnalysisInputError(
                "Every quantile probability must satisfy "
                "0 <= p <= 1."
            )

        probabilities.append(
            probability
        )

    if len(
        set(probabilities)
    ) != len(probabilities):
        raise InvalidAnalysisInputError(
            "'probabilities' cannot contain duplicates."
        )

    return probabilities


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
                "Advanced quantiles require genuinely "
                "numeric non-missing values."
            )

    numeric = np.asarray(
        observed,
        dtype=float,
    )

    if not np.all(
        np.isfinite(numeric)
    ):
        raise InvalidAnalysisInputError(
            "Advanced quantiles require finite "
            "non-missing values."
        )

    return numeric


class AdvancedQuantiles(
    AnalysisMethod
):
    name = "advanced_quantiles"
    category = "statistics"
    version = "1.0"

    description = (
        "Calculate configurable empirical quantiles, "
        "interquartile range, percentile range, and "
        "robust distribution summaries."
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
        values = kwargs.get("values")

        if values is None:
            raise InvalidAnalysisInputError(
                "'values' is required."
            )

        probabilities = (
            _validate_probabilities(
                kwargs.get(
                    "probabilities"
                )
            )
        )

        try:
            series = pd.Series(values)
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "'values' must be convertible to "
                "a one-dimensional series."
            ) from exc

        total_count = int(
            len(series)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Advanced quantiles require at least "
                "one observation."
            )

        missing_count = int(
            series.isna().sum()
        )

        observed = series.dropna()

        sample_size = int(
            len(observed)
        )

        if sample_size == 0:
            return {
                "statistic_type": "advanced_quantiles",
                "total_count": total_count,
                "sample_size": 0,
                "missing_count": missing_count,
                "is_all_missing": True,
                "probabilities": probabilities,
                "quantiles": {},
                "minimum": None,
                "maximum": None,
                "median": None,
                "q1": None,
                "q3": None,
                "iqr": None,
                "p05_p95_range": None,
            }

        numeric = _numeric_values(
            observed
        )

        calculated = np.quantile(
            numeric,
            probabilities,
            method="linear",
        )

        quantiles = {
            str(probability): float(value)
            for probability, value
            in zip(
                probabilities,
                calculated,
            )
        }

        q1 = float(
            np.quantile(
                numeric,
                0.25,
                method="linear",
            )
        )

        median = float(
            np.quantile(
                numeric,
                0.50,
                method="linear",
            )
        )

        q3 = float(
            np.quantile(
                numeric,
                0.75,
                method="linear",
            )
        )

        p05 = float(
            np.quantile(
                numeric,
                0.05,
                method="linear",
            )
        )

        p95 = float(
            np.quantile(
                numeric,
                0.95,
                method="linear",
            )
        )

        return {
            "statistic_type": "advanced_quantiles",
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "is_all_missing": False,
            "probabilities": probabilities,
            "quantiles": quantiles,
            "minimum": float(
                np.min(numeric)
            ),
            "maximum": float(
                np.max(numeric)
            ),
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": float(
                q3 - q1
            ),
            "p05_p95_range": float(
                p95 - p05
            ),
        }
