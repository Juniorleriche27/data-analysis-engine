import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.inference.parametric_tests import (
    _p_value,
    _student_t_cdf,
)
from data_analysis_engine.engines.regression.ols import (
    _ensure_numeric,
    _validate_boolean,
    _validate_columns,
)


_ALLOWED_COVARIANCE_TYPES = {
    "classical",
    "hc0",
    "hc1",
}


def _validate_covariance_type(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'covariance_type' must be a string."
        )

    covariance_type = (
        value.strip().lower()
    )

    if (
        covariance_type
        not in _ALLOWED_COVARIANCE_TYPES
    ):
        raise InvalidAnalysisInputError(
            "'covariance_type' must be one of: "
            "classical, hc0, hc1."
        )

    return covariance_type


def _validate_confidence_level(
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
            "'confidence_level' must be numeric."
        )

    confidence_level = float(
        value
    )

    if (
        not np.isfinite(
            confidence_level
        )
        or confidence_level <= 0.0
        or confidence_level >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'confidence_level' must satisfy "
            "0 < confidence_level < 1."
        )

    return confidence_level


def _student_t_quantile(
    probability: float,
    degrees_of_freedom: float,
) -> float:
    if (
        probability <= 0.0
        or probability >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "Student-t quantile probability must "
            "be strictly between zero and one."
        )

    if probability == 0.5:
        return 0.0

    if probability < 0.5:
        return float(
            -_student_t_quantile(
                1.0 - probability,
                degrees_of_freedom,
            )
        )

    lower = 0.0
    upper = 1.0

    while (
        _student_t_cdf(
            upper,
            degrees_of_freedom,
        )
        < probability
    ):
        upper *= 2.0

        if upper > 1_000_000.0:
            raise InvalidAnalysisInputError(
                "Unable to bracket Student-t quantile."
            )

    for _ in range(
        100
    ):
        midpoint = (
            lower
            + upper
        ) / 2.0

        cdf = _student_t_cdf(
            midpoint,
            degrees_of_freedom,
        )

        if cdf < probability:
            lower = midpoint
        else:
            upper = midpoint

    return float(
        (
            lower
            + upper
        )
        / 2.0
    )


class OrdinaryLeastSquaresInference(
    AnalysisMethod
):
    name = "ordinary_least_squares_inference"
    category = "regression"
    version = "1.0"

    description = (
        "Fit ordinary least squares regression and "
        "estimate coefficient uncertainty using classical, "
        "HC0, or HC1 covariance estimators with t statistics, "
        "p-values, and confidence intervals."
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
                "OLS inference requires at least "
                "two numerical columns."
            )

        try:
            frame = pd.DataFrame(
                data
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "All columns must form a rectangular "
                "dataframe."
            ) from exc

        total_count = int(
            len(frame)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "OLS inference requires observations."
            )

        target, predictors = _validate_columns(
            frame,
            kwargs.get(
                "target_column"
            ),
            kwargs.get(
                "predictor_columns"
            ),
        )

        fit_intercept = _validate_boolean(
            "fit_intercept",
            kwargs.get(
                "fit_intercept",
                True,
            ),
        )

        covariance_type = (
            _validate_covariance_type(
                kwargs.get(
                    "covariance_type",
                    "classical",
                )
            )
        )

        confidence_level = (
            _validate_confidence_level(
                kwargs.get(
                    "confidence_level",
                    0.95,
                )
            )
        )

        required_columns = [
            target,
            *predictors,
        ]

        numeric = _ensure_numeric(
            frame,
            required_columns,
        )

        complete_mask = ~numeric[
            required_columns
        ].isna().any(
            axis=1
        )

        complete = numeric.loc[
            complete_mask,
            required_columns,
        ]

        sample_size = int(
            len(complete)
        )

        dropped_count = int(
            total_count
            - sample_size
        )

        predictor_count = int(
            len(predictors)
        )

        parameter_count = int(
            predictor_count
            + (
                1
                if fit_intercept
                else 0
            )
        )

        if sample_size <= parameter_count:
            raise InvalidAnalysisInputError(
                "OLS inference requires more complete "
                "observations than fitted parameters."
            )

        y = complete[
            target
        ].to_numpy(
            dtype=float
        )

        x = complete[
            predictors
        ].to_numpy(
            dtype=float
        )

        if fit_intercept:
            design = np.column_stack(
                [
                    np.ones(
                        sample_size,
                        dtype=float,
                    ),
                    x,
                ]
            )

            parameter_names = [
                "intercept",
                *predictors,
            ]

        else:
            design = x

            parameter_names = list(
                predictors
            )

        matrix_rank = int(
            np.linalg.matrix_rank(
                design
            )
        )

        if matrix_rank < parameter_count:
            raise InvalidAnalysisInputError(
                "OLS design matrix is rank deficient. "
                "Remove constant or perfectly collinear "
                "predictors before inference."
            )

        coefficients, *_ = np.linalg.lstsq(
            design,
            y,
            rcond=None,
        )

        predictions = (
            design
            @ coefficients
        )

        residuals = (
            y
            - predictions
        )

        residual_sum_squares = float(
            np.sum(
                residuals ** 2
            )
        )

        residual_degrees_of_freedom = int(
            sample_size
            - parameter_count
        )

        xtx = (
            design.T
            @ design
        )

        try:
            xtx_inverse = np.linalg.inv(
                xtx
            )
        except np.linalg.LinAlgError as exc:
            raise InvalidAnalysisInputError(
                "Unable to invert the OLS information "
                "matrix."
            ) from exc

        if covariance_type == "classical":
            residual_variance = float(
                residual_sum_squares
                / residual_degrees_of_freedom
            )

            covariance_matrix = (
                residual_variance
                * xtx_inverse
            )

        else:
            weighted_design = (
                design
                * (
                    residuals ** 2
                )[:, None]
            )

            meat = (
                design.T
                @ weighted_design
            )

            covariance_matrix = (
                xtx_inverse
                @ meat
                @ xtx_inverse
            )

            if covariance_type == "hc1":
                covariance_matrix *= float(
                    sample_size
                    / residual_degrees_of_freedom
                )

            residual_variance = float(
                residual_sum_squares
                / residual_degrees_of_freedom
            )

        covariance_matrix = (
            covariance_matrix
            + covariance_matrix.T
        ) / 2.0

        diagonal = np.diag(
            covariance_matrix
        )

        if np.any(
            diagonal < -1e-12
        ):
            raise InvalidAnalysisInputError(
                "Estimated covariance matrix contains "
                "invalid negative variances."
            )

        diagonal = np.maximum(
            diagonal,
            0.0,
        )

        standard_errors = np.sqrt(
            diagonal
        )

        t_statistics: list[
            float | None
        ] = []

        infinite_statistics: list[
            bool
        ] = []

        p_values: list[
            float
        ] = []

        for coefficient, standard_error in zip(
            coefficients,
            standard_errors,
        ):
            if standard_error == 0.0:
                if coefficient == 0.0:
                    statistic = 0.0
                elif coefficient > 0.0:
                    statistic = float(
                        "inf"
                    )
                else:
                    statistic = float(
                        "-inf"
                    )
            else:
                statistic = float(
                    coefficient
                    / standard_error
                )

            infinite_statistics.append(
                bool(
                    np.isinf(
                        statistic
                    )
                )
            )

            t_statistics.append(
                None
                if not np.isfinite(
                    statistic
                )
                else statistic
            )

            p_values.append(
                float(
                    _p_value(
                        statistic,
                        float(
                            residual_degrees_of_freedom
                        ),
                        "two_sided",
                    )
                )
            )

        alpha = float(
            1.0
            - confidence_level
        )

        critical_probability = float(
            1.0
            - alpha / 2.0
        )

        critical_value = (
            _student_t_quantile(
                critical_probability,
                float(
                    residual_degrees_of_freedom
                ),
            )
        )

        coefficient_results: dict[
            str,
            dict[str, Any],
        ] = {}

        for index, name in enumerate(
            parameter_names
        ):
            coefficient = float(
                coefficients[
                    index
                ]
            )

            standard_error = float(
                standard_errors[
                    index
                ]
            )

            margin = float(
                critical_value
                * standard_error
            )

            coefficient_results[
                name
            ] = {
                "coefficient": (
                    coefficient
                ),
                "standard_error": (
                    standard_error
                ),
                "t_statistic": (
                    t_statistics[
                        index
                    ]
                ),
                "t_statistic_infinite": (
                    infinite_statistics[
                        index
                    ]
                ),
                "p_value": (
                    p_values[
                        index
                    ]
                ),
                "confidence_interval": {
                    "lower_bound": float(
                        coefficient
                        - margin
                    ),
                    "upper_bound": float(
                        coefficient
                        + margin
                    ),
                },
            }

        mean_y = float(
            np.mean(
                y
            )
        )

        total_sum_squares = float(
            np.sum(
                (
                    y
                    - mean_y
                )
                ** 2
            )
        )

        if total_sum_squares == 0.0:
            r_squared = (
                1.0
                if residual_sum_squares == 0.0
                else 0.0
            )
        else:
            r_squared = float(
                1.0
                - residual_sum_squares
                / total_sum_squares
            )

        if (
            fit_intercept
            and sample_size
            > predictor_count + 1
        ):
            adjusted_r_squared = float(
                1.0
                - (
                    1.0
                    - r_squared
                )
                * (
                    sample_size - 1
                )
                / (
                    sample_size
                    - predictor_count
                    - 1
                )
            )
        else:
            adjusted_r_squared = None

        covariance_serialized = {
            row_name: {
                column_name: float(
                    covariance_matrix[
                        row_index,
                        column_index,
                    ]
                )
                for column_index, column_name
                in enumerate(
                    parameter_names
                )
            }
            for row_index, row_name
            in enumerate(
                parameter_names
            )
        }

        return {
            "model_type": (
                "ordinary_least_squares_inference"
            ),
            "target_column": (
                target
            ),
            "predictor_columns": (
                predictors
            ),
            "fit_intercept": (
                fit_intercept
            ),
            "covariance_type": (
                covariance_type
            ),
            "confidence_level": (
                confidence_level
            ),
            "critical_value": (
                critical_value
            ),
            "total_count": (
                total_count
            ),
            "sample_size": (
                sample_size
            ),
            "dropped_count": (
                dropped_count
            ),
            "predictor_count": (
                predictor_count
            ),
            "parameter_count": (
                parameter_count
            ),
            "matrix_rank": (
                matrix_rank
            ),
            "residual_degrees_of_freedom": (
                residual_degrees_of_freedom
            ),
            "residual_variance": (
                residual_variance
            ),
            "residual_sum_squares": (
                residual_sum_squares
            ),
            "r_squared": (
                r_squared
            ),
            "adjusted_r_squared": (
                adjusted_r_squared
            ),
            "coefficients": (
                coefficient_results
            ),
            "covariance_matrix": (
                covariance_serialized
            ),
            "predictions": [
                float(value)
                for value
                in predictions
            ],
            "residuals": [
                float(value)
                for value
                in residuals
            ],
        }
