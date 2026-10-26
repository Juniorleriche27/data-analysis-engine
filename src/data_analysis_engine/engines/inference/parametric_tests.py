import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_ALLOWED_ALTERNATIVES = {
    "two_sided",
    "greater",
    "less",
}


# ============================================================
# Student-t distribution utilities
# Pure Python / math implementation.
# No SciPy dependency.
# ============================================================

def _beta_continued_fraction(
    a: float,
    b: float,
    x: float,
) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    fp_min = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0

    d = (
        1.0
        - qab
        * x
        / qap
    )

    if abs(d) < fp_min:
        d = fp_min

    d = 1.0 / d
    h = d

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        m2 = 2 * iteration

        aa = (
            iteration
            * (
                b - iteration
            )
            * x
            / (
                (
                    qam + m2
                )
                * (
                    a + m2
                )
            )
        )

        d = (
            1.0
            + aa
            * d
        )

        if abs(d) < fp_min:
            d = fp_min

        c = (
            1.0
            + aa
            / c
        )

        if abs(c) < fp_min:
            c = fp_min

        d = 1.0 / d

        h *= (
            d
            * c
        )

        aa = (
            -(
                a + iteration
            )
            * (
                qab + iteration
            )
            * x
            / (
                (
                    a + m2
                )
                * (
                    qap + m2
                )
            )
        )

        d = (
            1.0
            + aa
            * d
        )

        if abs(d) < fp_min:
            d = fp_min

        c = (
            1.0
            + aa
            / c
        )

        if abs(c) < fp_min:
            c = fp_min

        d = 1.0 / d

        delta = (
            d
            * c
        )

        h *= delta

        if abs(
            delta - 1.0
        ) <= epsilon:
            return float(h)

    raise InvalidAnalysisInputError(
        "Student-t probability calculation "
        "did not converge."
    )


def _regularized_incomplete_beta(
    x: float,
    a: float,
    b: float,
) -> float:
    if (
        a <= 0.0
        or b <= 0.0
    ):
        raise InvalidAnalysisInputError(
            "Beta-distribution parameters "
            "must be positive."
        )

    if x <= 0.0:
        return 0.0

    if x >= 1.0:
        return 1.0

    log_beta_term = (
        math.lgamma(
            a + b
        )
        - math.lgamma(a)
        - math.lgamma(b)
        + a
        * math.log(x)
        + b
        * math.log1p(
            -x
        )
    )

    beta_term = math.exp(
        log_beta_term
    )

    threshold = (
        a + 1.0
    ) / (
        a + b + 2.0
    )

    if x < threshold:
        value = (
            beta_term
            * _beta_continued_fraction(
                a,
                b,
                x,
            )
            / a
        )
    else:
        value = (
            1.0
            - beta_term
            * _beta_continued_fraction(
                b,
                a,
                1.0 - x,
            )
            / b
        )

    return float(
        min(
            max(
                value,
                0.0,
            ),
            1.0,
        )
    )


def _student_t_cdf(
    statistic: float,
    degrees_of_freedom: float,
) -> float:
    if (
        not np.isfinite(
            degrees_of_freedom
        )
        or degrees_of_freedom <= 0.0
    ):
        raise InvalidAnalysisInputError(
            "Degrees of freedom must be "
            "finite and positive."
        )

    if statistic == float("inf"):
        return 1.0

    if statistic == float("-inf"):
        return 0.0

    if statistic == 0.0:
        return 0.5

    x = float(
        degrees_of_freedom
        / (
            degrees_of_freedom
            + statistic ** 2
        )
    )

    beta_value = (
        _regularized_incomplete_beta(
            x,
            degrees_of_freedom / 2.0,
            0.5,
        )
    )

    if statistic > 0.0:
        cdf = (
            1.0
            - 0.5
            * beta_value
        )
    else:
        cdf = (
            0.5
            * beta_value
        )

    return float(
        min(
            max(
                cdf,
                0.0,
            ),
            1.0,
        )
    )


def _p_value(
    statistic: float,
    degrees_of_freedom: float,
    alternative: str,
) -> float:
    cdf = _student_t_cdf(
        statistic,
        degrees_of_freedom,
    )

    if alternative == "two_sided":
        value = (
            2.0
            * min(
                cdf,
                1.0 - cdf,
            )
        )

    elif alternative == "greater":
        value = (
            1.0 - cdf
        )

    else:
        value = cdf

    return float(
        min(
            max(
                value,
                0.0,
            ),
            1.0,
        )
    )


# ============================================================
# Validation helpers
# ============================================================

def _validate_alternative(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'alternative' must be a string."
        )

    alternative = (
        value.strip().lower()
    )

    if alternative not in _ALLOWED_ALTERNATIVES:
        raise InvalidAnalysisInputError(
            "'alternative' must be one of: "
            "two_sided, greater, less."
        )

    return alternative


def _validate_finite_number(
    name: str,
    value: Any,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be numeric."
        )

    result = float(value)

    if not np.isfinite(
        result
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be finite."
        )

    return result


def _numeric_sample(
    values: Any,
    label: str,
) -> tuple[
    np.ndarray,
    int,
    int,
]:
    try:
        series = pd.Series(
            values
        )
    except Exception as exc:
        raise InvalidAnalysisInputError(
            f"'{label}' must be convertible "
            "to a one-dimensional series."
        ) from exc

    total_count = int(
        len(series)
    )

    if total_count == 0:
        raise InvalidAnalysisInputError(
            f"'{label}' requires at least "
            "one observation."
        )

    missing_count = int(
        series.isna().sum()
    )

    observed = series.dropna()

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
                "Parametric mean tests require "
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
            "Parametric mean tests require finite "
            "non-missing values."
        )

    return (
        numeric,
        total_count,
        missing_count,
    )


def _resolve_two_columns(
    data: Any,
    requested_columns: Any,
) -> tuple[
    str,
    str,
]:
    if not isinstance(
        data,
        dict,
    ):
        raise InvalidAnalysisInputError(
            "'data' must be a dictionary "
            "of column names to values."
        )

    if len(data) < 2:
        raise InvalidAnalysisInputError(
            "This test requires two numerical columns."
        )

    available = list(
        data.keys()
    )

    if requested_columns is None:
        if len(available) != 2:
            raise InvalidAnalysisInputError(
                "When more than two columns are supplied, "
                "'columns' must specify exactly two names."
            )

        first = str(
            available[0]
        )

        second = str(
            available[1]
        )

    else:
        if (
            not isinstance(
                requested_columns,
                (list, tuple),
            )
            or len(
                requested_columns
            ) != 2
        ):
            raise InvalidAnalysisInputError(
                "'columns' must contain exactly "
                "two column names."
            )

        first = str(
            requested_columns[0]
        )

        second = str(
            requested_columns[1]
        )

    if first == second:
        raise InvalidAnalysisInputError(
            "Two distinct columns are required."
        )

    if (
        first not in data
        or second not in data
    ):
        raise InvalidAnalysisInputError(
            "One or more requested columns "
            "were not found."
        )

    return (
        first,
        second,
    )


# ============================================================
# One-sample t test
# ============================================================

class OneSampleMeanTTest(
    AnalysisMethod
):
    name = "one_sample_mean_t_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Perform a one-sample Student t-test for a "
        "population mean with two-sided or directional "
        "alternatives."
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

        hypothesized_mean = (
            _validate_finite_number(
                "hypothesized_mean",
                kwargs.get(
                    "hypothesized_mean",
                    0.0,
                ),
            )
        )

        alternative = (
            _validate_alternative(
                kwargs.get(
                    "alternative",
                    "two_sided",
                )
            )
        )

        numeric, total_count, missing_count = (
            _numeric_sample(
                values,
                "values",
            )
        )

        sample_size = int(
            len(numeric)
        )

        if sample_size < 2:
            raise InvalidAnalysisInputError(
                "One-sample t-testing requires at least "
                "two non-missing observations."
            )

        mean = float(
            np.mean(
                numeric
            )
        )

        standard_deviation = float(
            np.std(
                numeric,
                ddof=1,
            )
        )

        standard_error = float(
            standard_deviation
            / math.sqrt(
                sample_size
            )
        )

        mean_difference = float(
            mean
            - hypothesized_mean
        )

        degrees_of_freedom = float(
            sample_size - 1
        )

        if standard_error == 0.0:
            if mean_difference == 0.0:
                statistic = 0.0
            elif mean_difference > 0.0:
                statistic = float(
                    "inf"
                )
            else:
                statistic = float(
                    "-inf"
                )
        else:
            statistic = float(
                mean_difference
                / standard_error
            )

        p_value = _p_value(
            statistic,
            degrees_of_freedom,
            alternative,
        )

        standardized_effect = (
            None
            if standard_deviation == 0.0
            else float(
                mean_difference
                / standard_deviation
            )
        )

        return {
            "statistic_type": (
                "one_sample_mean_t_test"
            ),
            "alternative": alternative,
            "hypothesized_mean": (
                hypothesized_mean
            ),
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "sample_mean": mean,
            "mean_difference": (
                mean_difference
            ),
            "sample_standard_deviation": (
                standard_deviation
            ),
            "standard_error": (
                standard_error
            ),
            "degrees_of_freedom": (
                degrees_of_freedom
            ),
            "t_statistic": (
                None
                if not np.isfinite(
                    statistic
                )
                else statistic
            ),
            "t_statistic_infinite": bool(
                np.isinf(
                    statistic
                )
            ),
            "p_value": p_value,
            "standardized_effect": (
                standardized_effect
            ),
        }


# ============================================================
# Welch independent-samples t test
# ============================================================

class WelchMeanDifferenceTTest(
    AnalysisMethod
):
    name = "welch_mean_difference_t_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Compare two independent population means using "
        "Welch's unequal-variance t-test."
    )

    supported_roles = (
        "measure",
    )

    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = (
        (
            "measure",
            2,
        ),
    )

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = kwargs.get(
            "data"
        )

        first_name, second_name = (
            _resolve_two_columns(
                data,
                kwargs.get(
                    "columns"
                ),
            )
        )

        alternative = (
            _validate_alternative(
                kwargs.get(
                    "alternative",
                    "two_sided",
                )
            )
        )

        hypothesized_difference = (
            _validate_finite_number(
                "hypothesized_difference",
                kwargs.get(
                    "hypothesized_difference",
                    0.0,
                ),
            )
        )

        first, first_total, first_missing = (
            _numeric_sample(
                data[
                    first_name
                ],
                first_name,
            )
        )

        second, second_total, second_missing = (
            _numeric_sample(
                data[
                    second_name
                ],
                second_name,
            )
        )

        first_size = int(
            len(first)
        )

        second_size = int(
            len(second)
        )

        if (
            first_size < 2
            or second_size < 2
        ):
            raise InvalidAnalysisInputError(
                "Welch t-testing requires at least "
                "two non-missing observations "
                "in each sample."
            )

        first_mean = float(
            np.mean(first)
        )

        second_mean = float(
            np.mean(second)
        )

        first_variance = float(
            np.var(
                first,
                ddof=1,
            )
        )

        second_variance = float(
            np.var(
                second,
                ddof=1,
            )
        )

        observed_difference = float(
            first_mean
            - second_mean
        )

        adjusted_difference = float(
            observed_difference
            - hypothesized_difference
        )

        first_component = float(
            first_variance
            / first_size
        )

        second_component = float(
            second_variance
            / second_size
        )

        standard_error_squared = float(
            first_component
            + second_component
        )

        standard_error = float(
            math.sqrt(
                standard_error_squared
            )
        )

        if standard_error_squared == 0.0:
            degrees_of_freedom = float(
                first_size
                + second_size
                - 2
            )

            if adjusted_difference == 0.0:
                statistic = 0.0
            elif adjusted_difference > 0.0:
                statistic = float(
                    "inf"
                )
            else:
                statistic = float(
                    "-inf"
                )

        else:
            numerator = (
                standard_error_squared ** 2
            )

            denominator = (
                (
                    first_component ** 2
                )
                / (
                    first_size - 1
                )
                + (
                    second_component ** 2
                )
                / (
                    second_size - 1
                )
            )

            degrees_of_freedom = float(
                numerator
                / denominator
            )

            statistic = float(
                adjusted_difference
                / standard_error
            )

        p_value = _p_value(
            statistic,
            degrees_of_freedom,
            alternative,
        )

        pooled_variance = float(
            (
                (
                    first_size - 1
                )
                * first_variance
                + (
                    second_size - 1
                )
                * second_variance
            )
            / (
                first_size
                + second_size
                - 2
            )
        )

        pooled_standard_deviation = float(
            math.sqrt(
                pooled_variance
            )
        )

        standardized_effect = (
            None
            if pooled_standard_deviation == 0.0
            else float(
                observed_difference
                / pooled_standard_deviation
            )
        )

        return {
            "statistic_type": (
                "welch_mean_difference_t_test"
            ),
            "columns": [
                first_name,
                second_name,
            ],
            "alternative": alternative,
            "hypothesized_difference": (
                hypothesized_difference
            ),
            "first_total_count": (
                first_total
            ),
            "second_total_count": (
                second_total
            ),
            "first_sample_size": (
                first_size
            ),
            "second_sample_size": (
                second_size
            ),
            "first_missing_count": (
                first_missing
            ),
            "second_missing_count": (
                second_missing
            ),
            "first_mean": first_mean,
            "second_mean": second_mean,
            "observed_mean_difference": (
                observed_difference
            ),
            "adjusted_mean_difference": (
                adjusted_difference
            ),
            "first_variance": (
                first_variance
            ),
            "second_variance": (
                second_variance
            ),
            "standard_error": (
                standard_error
            ),
            "degrees_of_freedom": (
                degrees_of_freedom
            ),
            "t_statistic": (
                None
                if not np.isfinite(
                    statistic
                )
                else statistic
            ),
            "t_statistic_infinite": bool(
                np.isinf(
                    statistic
                )
            ),
            "p_value": p_value,
            "standardized_effect": (
                standardized_effect
            ),
        }


# ============================================================
# Paired-samples t test
# ============================================================

class PairedMeanDifferenceTTest(
    AnalysisMethod
):
    name = "paired_mean_difference_t_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Compare paired numerical observations using "
        "a paired Student t-test with pairwise "
        "missing-value exclusion."
    )

    supported_roles = (
        "measure",
    )

    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = (
        (
            "measure",
            2,
        ),
    )

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = kwargs.get(
            "data"
        )

        first_name, second_name = (
            _resolve_two_columns(
                data,
                kwargs.get(
                    "columns"
                ),
            )
        )

        alternative = (
            _validate_alternative(
                kwargs.get(
                    "alternative",
                    "two_sided",
                )
            )
        )

        hypothesized_difference = (
            _validate_finite_number(
                "hypothesized_difference",
                kwargs.get(
                    "hypothesized_difference",
                    0.0,
                ),
            )
        )

        try:
            first = pd.Series(
                data[
                    first_name
                ]
            )

            second = pd.Series(
                data[
                    second_name
                ]
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "Paired samples must be convertible "
                "to one-dimensional series."
            ) from exc

        if len(first) != len(second):
            raise InvalidAnalysisInputError(
                "Paired samples must contain the same "
                "number of observations."
            )

        total_pair_count = int(
            len(first)
        )

        if total_pair_count == 0:
            raise InvalidAnalysisInputError(
                "Paired t-testing requires at least "
                "one pair."
            )

        valid_mask = (
            ~first.isna()
            & ~second.isna()
        )

        complete_pair_count = int(
            valid_mask.sum()
        )

        missing_pair_count = int(
            total_pair_count
            - complete_pair_count
        )

        if complete_pair_count < 2:
            raise InvalidAnalysisInputError(
                "Paired t-testing requires at least "
                "two complete pairs."
            )

        first_values = first[
            valid_mask
        ]

        second_values = second[
            valid_mask
        ]

        first_numeric, _, _ = (
            _numeric_sample(
                first_values,
                first_name,
            )
        )

        second_numeric, _, _ = (
            _numeric_sample(
                second_values,
                second_name,
            )
        )

        differences = (
            first_numeric
            - second_numeric
        )

        mean_difference = float(
            np.mean(
                differences
            )
        )

        adjusted_difference = float(
            mean_difference
            - hypothesized_difference
        )

        difference_standard_deviation = float(
            np.std(
                differences,
                ddof=1,
            )
        )

        standard_error = float(
            difference_standard_deviation
            / math.sqrt(
                complete_pair_count
            )
        )

        degrees_of_freedom = float(
            complete_pair_count - 1
        )

        if standard_error == 0.0:
            if adjusted_difference == 0.0:
                statistic = 0.0
            elif adjusted_difference > 0.0:
                statistic = float(
                    "inf"
                )
            else:
                statistic = float(
                    "-inf"
                )
        else:
            statistic = float(
                adjusted_difference
                / standard_error
            )

        p_value = _p_value(
            statistic,
            degrees_of_freedom,
            alternative,
        )

        standardized_effect = (
            None
            if difference_standard_deviation == 0.0
            else float(
                mean_difference
                / difference_standard_deviation
            )
        )

        return {
            "statistic_type": (
                "paired_mean_difference_t_test"
            ),
            "columns": [
                first_name,
                second_name,
            ],
            "alternative": alternative,
            "hypothesized_difference": (
                hypothesized_difference
            ),
            "total_pair_count": (
                total_pair_count
            ),
            "complete_pair_count": (
                complete_pair_count
            ),
            "missing_pair_count": (
                missing_pair_count
            ),
            "first_mean": float(
                np.mean(
                    first_numeric
                )
            ),
            "second_mean": float(
                np.mean(
                    second_numeric
                )
            ),
            "observed_mean_difference": (
                mean_difference
            ),
            "adjusted_mean_difference": (
                adjusted_difference
            ),
            "difference_standard_deviation": (
                difference_standard_deviation
            ),
            "standard_error": (
                standard_error
            ),
            "degrees_of_freedom": (
                degrees_of_freedom
            ),
            "t_statistic": (
                None
                if not np.isfinite(
                    statistic
                )
                else statistic
            ),
            "t_statistic_infinite": bool(
                np.isinf(
                    statistic
                )
            ),
            "p_value": p_value,
            "standardized_effect": (
                standardized_effect
            ),
        }
