from numbers import Real
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_coverage(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'coverage' must be numeric."
        )

    coverage = float(value)

    if (
        not np.isfinite(coverage)
        or coverage <= 0.0
        or coverage >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'coverage' must satisfy 0 < coverage < 1."
        )

    return coverage


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
                "Mean confidence intervals require "
                "genuinely numeric non-missing values."
            )

    numeric = np.asarray(
        observed,
        dtype=float,
    )

    if not np.all(
        np.isfinite(numeric)
    ):
        raise InvalidAnalysisInputError(
            "Mean confidence intervals require finite "
            "non-missing values."
        )

    return numeric


class MeanConfidenceInterval(
    AnalysisMethod
):
    name = "mean_confidence_interval"
    category = "statistics"
    version = "1.0"

    description = (
        "Estimate a two-sided confidence interval for "
        "the population mean using the sample standard "
        "error and normal approximation."
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

        coverage = _validate_coverage(
            kwargs.get(
                "coverage",
                0.95,
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
                "Mean confidence intervals require "
                "at least one observation."
            )

        missing_count = int(
            series.isna().sum()
        )

        observed = series.dropna()

        sample_size = int(
            len(observed)
        )

        if sample_size < 2:
            raise InvalidAnalysisInputError(
                "Mean confidence intervals require "
                "at least two non-missing observations."
            )

        numeric = _numeric_values(
            observed
        )

        mean = float(
            np.mean(numeric)
        )

        sample_standard_deviation = float(
            np.std(
                numeric,
                ddof=1,
            )
        )

        standard_error = float(
            sample_standard_deviation
            / np.sqrt(
                sample_size
            )
        )

        probability = float(
            0.5
            + coverage / 2.0
        )

        critical_value = float(
            NormalDist().inv_cdf(
                probability
            )
        )

        margin_of_error = float(
            critical_value
            * standard_error
        )

        lower_bound = float(
            mean
            - margin_of_error
        )

        upper_bound = float(
            mean
            + margin_of_error
        )

        return {
            "statistic_type": (
                "mean_confidence_interval"
            ),
            "method": (
                "normal_approximation"
            ),
            "coverage": coverage,
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "mean": mean,
            "sample_standard_deviation": (
                sample_standard_deviation
            ),
            "standard_error": (
                standard_error
            ),
            "critical_value": (
                critical_value
            ),
            "margin_of_error": (
                margin_of_error
            ),
            "lower_bound": (
                lower_bound
            ),
            "upper_bound": (
                upper_bound
            ),
            "interval_width": float(
                upper_bound
                - lower_bound
            ),
            "is_constant": bool(
                sample_standard_deviation
                == 0.0
            ),
            "assumption_note": (
                "This interval uses a normal approximation "
                "with the sample standard error."
            ),
        }
