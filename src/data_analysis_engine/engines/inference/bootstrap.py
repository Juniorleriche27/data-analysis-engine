from numbers import Integral, Real
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


def _validate_n_resamples(
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise InvalidAnalysisInputError(
            "'n_resamples' must be an integer."
        )

    n_resamples = int(value)

    if n_resamples < 100:
        raise InvalidAnalysisInputError(
            "'n_resamples' must be at least 100."
        )

    return n_resamples


def _validate_random_state(
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise InvalidAnalysisInputError(
            "'random_state' must be an integer."
        )

    return int(value)


def _numeric_values(
    observed: pd.Series,
) -> np.ndarray:
    for value in observed:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
        ):
            raise InvalidAnalysisInputError(
                "Bootstrap inference requires genuinely "
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
            "Bootstrap inference requires finite "
            "non-missing values."
        )

    return numeric


class BootstrapMeanConfidenceInterval(
    AnalysisMethod
):
    name = "bootstrap_mean_confidence_interval"
    category = "statistics"
    version = "1.0"

    description = (
        "Estimate a reproducible percentile bootstrap "
        "confidence interval for a population mean."
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

        coverage = _validate_coverage(
            kwargs.get(
                "coverage",
                0.95,
            )
        )

        n_resamples = _validate_n_resamples(
            kwargs.get(
                "n_resamples",
                2000,
            )
        )

        random_state = _validate_random_state(
            kwargs.get(
                "random_state",
                42,
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
                "Bootstrap inference requires "
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
                "Bootstrap mean inference requires "
                "at least two non-missing observations."
            )

        numeric = _numeric_values(
            observed
        )

        observed_mean = float(
            np.mean(numeric)
        )

        generator = np.random.default_rng(
            random_state
        )

        indices = generator.integers(
            0,
            sample_size,
            size=(
                n_resamples,
                sample_size,
            ),
        )

        bootstrap_samples = numeric[
            indices
        ]

        bootstrap_means = np.mean(
            bootstrap_samples,
            axis=1,
        )

        alpha = float(
            1.0 - coverage
        )

        lower_probability = float(
            alpha / 2.0
        )

        upper_probability = float(
            1.0 - alpha / 2.0
        )

        lower_bound = float(
            np.quantile(
                bootstrap_means,
                lower_probability,
                method="linear",
            )
        )

        upper_bound = float(
            np.quantile(
                bootstrap_means,
                upper_probability,
                method="linear",
            )
        )

        bootstrap_standard_error = float(
            np.std(
                bootstrap_means,
                ddof=1,
            )
        )

        bootstrap_bias = float(
            np.mean(
                bootstrap_means
            )
            - observed_mean
        )

        return {
            "statistic_type": (
                "bootstrap_mean_confidence_interval"
            ),
            "method": "percentile_bootstrap",
            "coverage": coverage,
            "n_resamples": n_resamples,
            "random_state": random_state,
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "observed_mean": observed_mean,
            "bootstrap_mean": float(
                np.mean(
                    bootstrap_means
                )
            ),
            "bootstrap_bias": (
                bootstrap_bias
            ),
            "bootstrap_standard_error": (
                bootstrap_standard_error
            ),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width": float(
                upper_bound
                - lower_bound
            ),
            "is_constant": bool(
                np.ptp(numeric)
                == 0.0
            ),
        }
