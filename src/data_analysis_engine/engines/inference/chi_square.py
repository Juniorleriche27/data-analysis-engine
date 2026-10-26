import math
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _regularized_gamma_q(
    a: float,
    x: float,
) -> float:
    if (
        a <= 0.0
        or x < 0.0
    ):
        raise InvalidAnalysisInputError(
            "Invalid gamma-function parameters."
        )

    if x == 0.0:
        return 1.0

    epsilon = 3.0e-14
    fp_min = 1.0e-300
    max_iterations = 300

    log_factor = (
        -x
        + a
        * math.log(x)
        - math.lgamma(a)
    )

    if x < (
        a + 1.0
    ):
        ap = a
        term = (
            1.0
            / a
        )
        total = term

        for _ in range(
            max_iterations
        ):
            ap += 1.0

            term *= (
                x
                / ap
            )

            total += term

            if abs(term) <= (
                abs(total)
                * epsilon
            ):
                p_value = float(
                    total
                    * math.exp(
                        log_factor
                    )
                )

                return float(
                    min(
                        max(
                            1.0 - p_value,
                            0.0,
                        ),
                        1.0,
                    )
                )

        raise InvalidAnalysisInputError(
            "Chi-square probability calculation "
            "did not converge."
        )

    b = (
        x
        + 1.0
        - a
    )

    c = (
        1.0
        / fp_min
    )

    d = (
        1.0
        / b
    )

    h = d

    for iteration in range(
        1,
        max_iterations + 1,
    ):
        an = float(
            -iteration
            * (
                iteration - a
            )
        )

        b += 2.0

        d = (
            an
            * d
            + b
        )

        if abs(d) < fp_min:
            d = fp_min

        c = (
            b
            + an
            / c
        )

        if abs(c) < fp_min:
            c = fp_min

        d = (
            1.0
            / d
        )

        delta = (
            d
            * c
        )

        h *= delta

        if abs(
            delta - 1.0
        ) <= epsilon:
            q_value = float(
                math.exp(
                    log_factor
                )
                * h
            )

            return float(
                min(
                    max(
                        q_value,
                        0.0,
                    ),
                    1.0,
                )
            )

    raise InvalidAnalysisInputError(
        "Chi-square probability calculation "
        "did not converge."
    )


def _python_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return value


class ChiSquareIndependenceTest(
    AnalysisMethod
):
    name = "chi_square_independence_test"
    category = "statistics"
    version = "1.0"

    description = (
        "Test independence between two categorical "
        "variables using Pearson's chi-square statistic "
        "and report Cramer's V effect size."
    )

    supported_roles = (
        "dimension",
        "text",
    )

    minimum_compatible_columns = 2
    selection_mode = "joint"
    analysis_unit = "dataset"
    role_requirements = ()

    allow_constant_columns = False
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
                "Chi-square independence testing "
                "requires two categorical columns."
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
                "Categorical variables must be convertible "
                "to one-dimensional series."
            ) from exc

        if len(first) != len(second):
            raise InvalidAnalysisInputError(
                "Chi-square variables must contain "
                "the same number of observations."
            )

        total_count = int(
            len(first)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Chi-square independence testing "
                "requires observations."
            )

        valid_mask = (
            ~first.isna()
            & ~second.isna()
        )

        complete_count = int(
            valid_mask.sum()
        )

        missing_pair_count = int(
            total_count
            - complete_count
        )

        if complete_count == 0:
            raise InvalidAnalysisInputError(
                "Chi-square independence testing requires "
                "at least one complete observation pair."
            )

        complete_first = first[
            valid_mask
        ]

        complete_second = second[
            valid_mask
        ]

        contingency = pd.crosstab(
            complete_first,
            complete_second,
            dropna=False,
        )

        row_count = int(
            contingency.shape[0]
        )

        column_count = int(
            contingency.shape[1]
        )

        if (
            row_count < 2
            or column_count < 2
        ):
            raise InvalidAnalysisInputError(
                "Chi-square independence testing requires "
                "at least two observed categories "
                "in each variable."
            )

        observed = contingency.to_numpy(
            dtype=float
        )

        row_totals = np.sum(
            observed,
            axis=1,
        )

        column_totals = np.sum(
            observed,
            axis=0,
        )

        grand_total = float(
            np.sum(
                observed
            )
        )

        expected = (
            np.outer(
                row_totals,
                column_totals,
            )
            / grand_total
        )

        if np.any(
            expected <= 0.0
        ):
            raise InvalidAnalysisInputError(
                "Expected contingency counts "
                "must be positive."
            )

        chi_square = float(
            np.sum(
                (
                    observed
                    - expected
                )
                ** 2
                / expected
            )
        )

        degrees_of_freedom = int(
            (
                row_count - 1
            )
            * (
                column_count - 1
            )
        )

        p_value = (
            1.0
            if chi_square == 0.0
            else _regularized_gamma_q(
                degrees_of_freedom
                / 2.0,
                chi_square
                / 2.0,
            )
        )

        denominator_dimension = min(
            row_count - 1,
            column_count - 1,
        )

        cramers_v = float(
            math.sqrt(
                chi_square
                / (
                    grand_total
                    * denominator_dimension
                )
            )
        )

        expected_below_five_count = int(
            np.sum(
                expected < 5.0
            )
        )

        expected_below_one_count = int(
            np.sum(
                expected < 1.0
            )
        )

        expected_cell_count = int(
            expected.size
        )

        expected_below_five_rate = float(
            expected_below_five_count
            / expected_cell_count
        )

        assumptions_warning = bool(
            expected_below_one_count > 0
            or expected_below_five_rate > 0.20
        )

        return {
            "statistic_type": (
                "chi_square_independence_test"
            ),
            "columns": [
                first_name,
                second_name,
            ],
            "total_count": total_count,
            "complete_pair_count": (
                complete_count
            ),
            "missing_pair_count": (
                missing_pair_count
            ),
            "row_category_count": (
                row_count
            ),
            "column_category_count": (
                column_count
            ),
            "row_categories": [
                _python_scalar(
                    value
                )
                for value
                in contingency.index
            ],
            "column_categories": [
                _python_scalar(
                    value
                )
                for value
                in contingency.columns
            ],
            "observed_counts": (
                observed.astype(
                    int
                ).tolist()
            ),
            "expected_counts": (
                expected.tolist()
            ),
            "chi_square_statistic": (
                chi_square
            ),
            "degrees_of_freedom": (
                degrees_of_freedom
            ),
            "p_value": float(
                p_value
            ),
            "cramers_v": (
                cramers_v
            ),
            "minimum_expected_count": float(
                np.min(
                    expected
                )
            ),
            "expected_below_five_count": (
                expected_below_five_count
            ),
            "expected_below_five_rate": (
                expected_below_five_rate
            ),
            "expected_below_one_count": (
                expected_below_one_count
            ),
            "assumptions_warning": (
                assumptions_warning
            ),
        }
