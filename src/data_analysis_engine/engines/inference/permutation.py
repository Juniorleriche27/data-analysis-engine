from numbers import Integral, Real
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


def _numeric_array(
    values: Any,
    label: str,
) -> tuple[np.ndarray, int, int]:
    try:
        series = pd.Series(
            values
        )
    except Exception as exc:
        raise InvalidAnalysisInputError(
            f"Column '{label}' cannot be converted "
            "to a one-dimensional series."
        ) from exc

    total_count = int(
        len(series)
    )

    missing_count = int(
        series.isna().sum()
    )

    observed = series.dropna()

    if len(observed) == 0:
        raise InvalidAnalysisInputError(
            f"Column '{label}' requires at least "
            "one non-missing value."
        )

    for value in observed:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
        ):
            raise InvalidAnalysisInputError(
                "Permutation inference requires genuinely "
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
            "Permutation inference requires finite "
            "non-missing values."
        )

    return (
        numeric,
        total_count,
        missing_count,
    )


class PermutationMeanDifferenceTest(
    AnalysisMethod
):
    name = "permutation_mean_difference_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Test a difference in independent-sample means "
        "using a reproducible Monte Carlo permutation test."
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
                "Permutation mean-difference testing "
                "requires two numerical columns."
            )

        requested_columns = kwargs.get(
            "columns"
        )

        available_columns = list(
            data.keys()
        )

        if requested_columns is None:
            if len(
                available_columns
            ) != 2:
                raise InvalidAnalysisInputError(
                    "When more than two columns are supplied, "
                    "'columns' must specify exactly two names."
                )

            columns = [
                available_columns[0],
                available_columns[1],
            ]

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

            columns = [
                str(
                    requested_columns[0]
                ),
                str(
                    requested_columns[1]
                ),
            ]

            if columns[0] == columns[1]:
                raise InvalidAnalysisInputError(
                    "'columns' must reference "
                    "two distinct columns."
                )

            for column in columns:
                if column not in data:
                    raise InvalidAnalysisInputError(
                        f"Column '{column}' was not found."
                    )

        alternative = _validate_alternative(
            kwargs.get(
                "alternative",
                "two_sided",
            )
        )

        n_resamples = _validate_n_resamples(
            kwargs.get(
                "n_resamples",
                5000,
            )
        )

        random_state = _validate_random_state(
            kwargs.get(
                "random_state",
                42,
            )
        )

        first, first_total, first_missing = (
            _numeric_array(
                data[
                    columns[0]
                ],
                columns[0],
            )
        )

        second, second_total, second_missing = (
            _numeric_array(
                data[
                    columns[1]
                ],
                columns[1],
            )
        )

        first_mean = float(
            np.mean(first)
        )

        second_mean = float(
            np.mean(second)
        )

        observed_difference = float(
            first_mean
            - second_mean
        )

        pooled = np.concatenate(
            [
                first,
                second,
            ]
        )

        first_size = int(
            len(first)
        )

        second_size = int(
            len(second)
        )

        generator = np.random.default_rng(
            random_state
        )

        permutation_differences = np.empty(
            n_resamples,
            dtype=float,
        )

        for index in range(
            n_resamples
        ):
            permutation = generator.permutation(
                pooled
            )

            permutation_differences[
                index
            ] = float(
                np.mean(
                    permutation[
                        :first_size
                    ]
                )
                - np.mean(
                    permutation[
                        first_size:
                    ]
                )
            )

        tolerance = 1e-15

        if alternative == "two_sided":
            extreme_count = int(
                np.sum(
                    np.abs(
                        permutation_differences
                    )
                    >= (
                        abs(
                            observed_difference
                        )
                        - tolerance
                    )
                )
            )

        elif alternative == "greater":
            extreme_count = int(
                np.sum(
                    permutation_differences
                    >= (
                        observed_difference
                        - tolerance
                    )
                )
            )

        else:
            extreme_count = int(
                np.sum(
                    permutation_differences
                    <= (
                        observed_difference
                        + tolerance
                    )
                )
            )

        p_value = float(
            (
                extreme_count
                + 1
            )
            / (
                n_resamples
                + 1
            )
        )

        pooled_standard_deviation = float(
            np.std(
                pooled,
                ddof=1,
            )
        ) if len(pooled) > 1 else 0.0

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
                "permutation_mean_difference_test"
            ),
            "method": (
                "monte_carlo_permutation"
            ),
            "columns": columns,
            "alternative": alternative,
            "n_resamples": n_resamples,
            "random_state": random_state,
            "first_sample_size": (
                first_size
            ),
            "second_sample_size": (
                second_size
            ),
            "first_total_count": (
                first_total
            ),
            "second_total_count": (
                second_total
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
            "permutation_mean_difference": float(
                np.mean(
                    permutation_differences
                )
            ),
            "permutation_standard_deviation": float(
                np.std(
                    permutation_differences,
                    ddof=1,
                )
            ),
            "extreme_count": extreme_count,
            "p_value": p_value,
            "pooled_standard_deviation": (
                pooled_standard_deviation
            ),
            "standardized_effect": (
                standardized_effect
            ),
        }
