import math
from numbers import Real
from statistics import NormalDist
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

    alternative = value.strip().lower()

    if alternative not in _ALLOWED_ALTERNATIVES:
        raise InvalidAnalysisInputError(
            "'alternative' must be one of: "
            "two_sided, greater, less."
        )

    return alternative


def _validate_probability(
    name: str,
    value: Any,
    *,
    strict: bool,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be numeric."
        )

    probability = float(value)

    if not np.isfinite(
        probability
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be finite."
        )

    if strict:
        valid = (
            0.0
            < probability
            < 1.0
        )
    else:
        valid = (
            0.0
            <= probability
            <= 1.0
        )

    if not valid:
        operator = (
            "0 < p < 1"
            if strict
            else "0 <= p <= 1"
        )

        raise InvalidAnalysisInputError(
            f"'{name}' must satisfy {operator}."
        )

    return probability


def _validate_success_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        (
            list,
            dict,
            set,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
        ),
    ):
        raise InvalidAnalysisInputError(
            "'success_value' must be a scalar."
        )

    try:
        missing = bool(
            pd.isna(value)
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise InvalidAnalysisInputError(
            "'success_value' must be a scalar."
        ) from exc

    if missing:
        raise InvalidAnalysisInputError(
            "'success_value' cannot be missing."
        )

    return value


def _normal_p_value(
    statistic: float,
    alternative: str,
) -> float:
    if statistic == float(
        "inf"
    ):
        cdf = 1.0

    elif statistic == float(
        "-inf"
    ):
        cdf = 0.0

    else:
        cdf = float(
            NormalDist().cdf(
                statistic
            )
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


def _sample_proportion(
    values: Any,
    success_value: Any,
    label: str,
) -> dict[str, Any]:
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

    missing_mask = series.isna()

    missing_count = int(
        missing_mask.sum()
    )

    observed = series[
        ~missing_mask
    ]

    sample_size = int(
        len(observed)
    )

    if sample_size == 0:
        raise InvalidAnalysisInputError(
            f"'{label}' requires at least "
            "one non-missing observation."
        )

    try:
        successes = np.asarray(
            [
                bool(
                    value
                    == success_value
                )
                for value
                in observed
            ],
            dtype=bool,
        )
    except Exception as exc:
        raise InvalidAnalysisInputError(
            "Observed values must support scalar "
            "equality comparison."
        ) from exc

    success_count = int(
        np.sum(
            successes
        )
    )

    proportion = float(
        success_count
        / sample_size
    )

    return {
        "total_count": total_count,
        "missing_count": missing_count,
        "sample_size": sample_size,
        "success_count": success_count,
        "failure_count": int(
            sample_size
            - success_count
        ),
        "proportion": proportion,
    }


class OneSampleProportionZTest(
    AnalysisMethod
):
    name = "one_sample_proportion_z_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Test a single observed success proportion "
        "against a hypothesized population proportion "
        "using a normal z approximation."
    )

    supported_roles = (
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

        success_value = (
            _validate_success_value(
                kwargs.get(
                    "success_value",
                    1,
                )
            )
        )

        hypothesized_proportion = (
            _validate_probability(
                "hypothesized_proportion",
                kwargs.get(
                    "hypothesized_proportion",
                    0.5,
                ),
                strict=True,
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

        sample = _sample_proportion(
            values,
            success_value,
            "values",
        )

        sample_size = sample[
            "sample_size"
        ]

        observed_proportion = sample[
            "proportion"
        ]

        difference = float(
            observed_proportion
            - hypothesized_proportion
        )

        standard_error = float(
            math.sqrt(
                hypothesized_proportion
                * (
                    1.0
                    - hypothesized_proportion
                )
                / sample_size
            )
        )

        statistic = float(
            difference
            / standard_error
        )

        p_value = _normal_p_value(
            statistic,
            alternative,
        )

        return {
            "statistic_type": (
                "one_sample_proportion_z_test"
            ),
            "alternative": alternative,
            "success_value": (
                success_value
            ),
            "hypothesized_proportion": (
                hypothesized_proportion
            ),
            **sample,
            "observed_proportion": (
                observed_proportion
            ),
            "proportion_difference": (
                difference
            ),
            "standard_error": (
                standard_error
            ),
            "z_statistic": statistic,
            "p_value": p_value,
        }


class TwoSampleProportionZTest(
    AnalysisMethod
):
    name = "two_sample_proportion_z_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Compare two independent success proportions "
        "using the pooled two-sample proportion z-test."
    )

    supported_roles = (
        "measure",
        "dimension",
        "text",
    )

    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "dataset"
    role_requirements = ()

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
                "Two-sample proportion testing "
                "requires two columns."
            )

        requested_columns = kwargs.get(
            "columns"
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

            first_name = str(
                available[0]
            )

            second_name = str(
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

            first_name = str(
                requested_columns[0]
            )

            second_name = str(
                requested_columns[1]
            )

        if first_name == second_name:
            raise InvalidAnalysisInputError(
                "Two distinct columns are required."
            )

        if (
            first_name not in data
            or second_name not in data
        ):
            raise InvalidAnalysisInputError(
                "One or more requested columns "
                "were not found."
            )

        success_value = (
            _validate_success_value(
                kwargs.get(
                    "success_value",
                    1,
                )
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

        first = _sample_proportion(
            data[
                first_name
            ],
            success_value,
            first_name,
        )

        second = _sample_proportion(
            data[
                second_name
            ],
            success_value,
            second_name,
        )

        first_size = first[
            "sample_size"
        ]

        second_size = second[
            "sample_size"
        ]

        first_successes = first[
            "success_count"
        ]

        second_successes = second[
            "success_count"
        ]

        first_proportion = first[
            "proportion"
        ]

        second_proportion = second[
            "proportion"
        ]

        observed_difference = float(
            first_proportion
            - second_proportion
        )

        pooled_proportion = float(
            (
                first_successes
                + second_successes
            )
            / (
                first_size
                + second_size
            )
        )

        standard_error_squared = float(
            pooled_proportion
            * (
                1.0
                - pooled_proportion
            )
            * (
                1.0
                / first_size
                + 1.0
                / second_size
            )
        )

        standard_error = float(
            math.sqrt(
                standard_error_squared
            )
        )

        if standard_error == 0.0:
            if observed_difference == 0.0:
                statistic = 0.0
            elif observed_difference > 0.0:
                statistic = float(
                    "inf"
                )
            else:
                statistic = float(
                    "-inf"
                )
        else:
            statistic = float(
                observed_difference
                / standard_error
            )

        p_value = _normal_p_value(
            statistic,
            alternative,
        )

        unpooled_standard_error = float(
            math.sqrt(
                first_proportion
                * (
                    1.0
                    - first_proportion
                )
                / first_size
                + second_proportion
                * (
                    1.0
                    - second_proportion
                )
                / second_size
            )
        )

        return {
            "statistic_type": (
                "two_sample_proportion_z_test"
            ),
            "columns": [
                first_name,
                second_name,
            ],
            "alternative": alternative,
            "success_value": (
                success_value
            ),
            "first_total_count": (
                first[
                    "total_count"
                ]
            ),
            "second_total_count": (
                second[
                    "total_count"
                ]
            ),
            "first_sample_size": (
                first_size
            ),
            "second_sample_size": (
                second_size
            ),
            "first_missing_count": (
                first[
                    "missing_count"
                ]
            ),
            "second_missing_count": (
                second[
                    "missing_count"
                ]
            ),
            "first_success_count": (
                first_successes
            ),
            "second_success_count": (
                second_successes
            ),
            "first_proportion": (
                first_proportion
            ),
            "second_proportion": (
                second_proportion
            ),
            "observed_proportion_difference": (
                observed_difference
            ),
            "pooled_proportion": (
                pooled_proportion
            ),
            "standard_error": (
                standard_error
            ),
            "unpooled_standard_error": (
                unpooled_standard_error
            ),
            "z_statistic": (
                None
                if not np.isfinite(
                    statistic
                )
                else statistic
            ),
            "z_statistic_infinite": bool(
                np.isinf(
                    statistic
                )
            ),
            "p_value": p_value,
        }
