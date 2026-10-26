import math
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.inference.chi_square import (
    _regularized_gamma_q,
)
from data_analysis_engine.engines.inference.parametric_tests import (
    _p_value,
    _regularized_incomplete_beta,
)
from data_analysis_engine.engines.regression.diagnostics import (
    _prepare_ols,
)
from data_analysis_engine.engines.regression.ols import (
    _ensure_numeric,
    _validate_boolean,
)
from data_analysis_engine.engines.regression.ols_inference import (
    _student_t_quantile,
    _validate_confidence_level,
)


def _validate_lags(
    value: Any,
    sample_size: int,
) -> int:
    if value is None:
        automatic = int(
            math.floor(
                4.0
                * (
                    sample_size
                    / 100.0
                )
                ** (
                    2.0 / 9.0
                )
            )
        )

        return int(
            min(
                max(
                    automatic,
                    0,
                ),
                sample_size - 1,
            )
        )

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise InvalidAnalysisInputError(
            "'max_lag' must be an integer or None."
        )

    max_lag = int(value)

    if (
        max_lag < 0
        or max_lag >= sample_size
    ):
        raise InvalidAnalysisInputError(
            "'max_lag' must satisfy "
            "0 <= max_lag < sample_size."
        )

    return max_lag


def _validate_predictor_list(
    name: str,
    value: Any,
    *,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(
        value,
        (list, tuple),
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be a list or tuple."
        )

    predictors = [
        str(column)
        for column
        in value
    ]

    if (
        not allow_empty
        and not predictors
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must contain at least "
            "one predictor."
        )

    if len(
        set(
            predictors
        )
    ) != len(
        predictors
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' cannot contain duplicates."
        )

    return predictors


def _fit_design(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str],
    fit_intercept: bool,
) -> dict[str, Any]:
    y = frame[
        target
    ].to_numpy(
        dtype=float
    )

    sample_size = int(
        len(frame)
    )

    if predictors:
        x = frame[
            predictors
        ].to_numpy(
            dtype=float
        )
    else:
        x = np.empty(
            (
                sample_size,
                0,
            ),
            dtype=float,
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
    else:
        if not predictors:
            raise InvalidAnalysisInputError(
                "A model without an intercept requires "
                "at least one predictor."
            )

        design = x

    parameter_count = int(
        design.shape[1]
    )

    if sample_size <= parameter_count:
        raise InvalidAnalysisInputError(
            "OLS model requires more observations "
            "than fitted parameters."
        )

    rank = int(
        np.linalg.matrix_rank(
            design
        )
    )

    if rank < parameter_count:
        raise InvalidAnalysisInputError(
            "OLS design matrix is rank deficient."
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

    rss = float(
        np.sum(
            residuals ** 2
        )
    )

    return {
        "design": design,
        "coefficients": coefficients,
        "predictions": predictions,
        "residuals": residuals,
        "rss": rss,
        "parameter_count": parameter_count,
        "degrees_of_freedom": int(
            sample_size
            - parameter_count
        ),
    }


class NeweyWestOLSInference(
    AnalysisMethod
):
    name = "newey_west_ols_inference"
    category = "regression"
    version = "1.0"

    description = (
        "Fit OLS and estimate Bartlett-kernel Newey-West "
        "heteroskedasticity and autocorrelation consistent "
        "coefficient covariance, standard errors, p-values, "
        "and confidence intervals."
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
        confidence_level = (
            _validate_confidence_level(
                kwargs.get(
                    "confidence_level",
                    0.95,
                )
            )
        )

        small_sample_correction = (
            _validate_boolean(
                "small_sample_correction",
                kwargs.get(
                    "small_sample_correction",
                    True,
                ),
            )
        )

        fit = _prepare_ols(
            kwargs.get("data"),
            kwargs.get("target_column"),
            kwargs.get("predictor_columns"),
            kwargs.get(
                "fit_intercept",
                True,
            ),
        )

        design = fit[
            "design"
        ]

        residuals = fit[
            "residuals"
        ]

        coefficients = fit[
            "coefficients"
        ]

        sample_size = fit[
            "sample_size"
        ]

        parameter_count = fit[
            "parameter_count"
        ]

        degrees_of_freedom = fit[
            "residual_degrees_of_freedom"
        ]

        max_lag = _validate_lags(
            kwargs.get(
                "max_lag"
            ),
            sample_size,
        )

        xtx_inverse = np.linalg.inv(
            design.T
            @ design
        )

        score = (
            design
            * residuals[:, None]
        )

        meat = (
            score.T
            @ score
        )

        for lag in range(
            1,
            max_lag + 1,
        ):
            weight = float(
                1.0
                - lag
                / (
                    max_lag + 1.0
                )
            )

            gamma = (
                score[
                    lag:
                ].T
                @ score[
                    :-lag
                ]
            )

            meat += (
                weight
                * (
                    gamma
                    + gamma.T
                )
            )

        covariance = (
            xtx_inverse
            @ meat
            @ xtx_inverse
        )

        if small_sample_correction:
            covariance *= float(
                sample_size
                / degrees_of_freedom
            )

        covariance = (
            covariance
            + covariance.T
        ) / 2.0

        diagonal = np.diag(
            covariance
        )

        if np.any(
            diagonal < -1e-10
        ):
            raise InvalidAnalysisInputError(
                "Newey-West covariance produced "
                "invalid negative coefficient variances."
            )

        standard_errors = np.sqrt(
            np.maximum(
                diagonal,
                0.0,
            )
        )

        if fit[
            "fit_intercept"
        ]:
            parameter_names = [
                "intercept",
                *fit[
                    "predictors"
                ],
            ]
        else:
            parameter_names = list(
                fit[
                    "predictors"
                ]
            )

        alpha = float(
            1.0
            - confidence_level
        )

        critical_value = (
            _student_t_quantile(
                1.0
                - alpha / 2.0,
                float(
                    degrees_of_freedom
                ),
            )
        )

        results: dict[
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

            p_value = float(
                _p_value(
                    statistic,
                    float(
                        degrees_of_freedom
                    ),
                    "two_sided",
                )
            )

            margin = float(
                critical_value
                * standard_error
            )

            results[
                name
            ] = {
                "coefficient": coefficient,
                "standard_error": standard_error,
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

        covariance_serialized = {
            row_name: {
                column_name: float(
                    covariance[
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
                "newey_west_ols_inference"
            ),
            "target_column": fit[
                "target"
            ],
            "predictor_columns": fit[
                "predictors"
            ],
            "fit_intercept": fit[
                "fit_intercept"
            ],
            "covariance_type": (
                "newey_west_hac"
            ),
            "kernel": "bartlett",
            "max_lag": max_lag,
            "small_sample_correction": (
                small_sample_correction
            ),
            "confidence_level": (
                confidence_level
            ),
            "critical_value": (
                critical_value
            ),
            "sample_size": (
                sample_size
            ),
            "dropped_count": fit[
                "dropped_count"
            ],
            "parameter_count": (
                parameter_count
            ),
            "residual_degrees_of_freedom": (
                degrees_of_freedom
            ),
            "coefficients": results,
            "covariance_matrix": (
                covariance_serialized
            ),
        }


class NestedOLSFTest(
    AnalysisMethod
):
    name = "nested_ols_f_test"
    category = "regression"
    version = "1.0"

    description = (
        "Compare restricted and full nested OLS models "
        "with a classical partial F-test for the joint "
        "significance of added predictors."
    )

    supported_roles = (
        "measure",
    )

    minimum_compatible_columns = 3
    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = (
        (
            "measure",
            3,
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
                "'data' must be a dictionary."
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

        target = kwargs.get(
            "target_column"
        )

        if (
            not isinstance(
                target,
                str,
            )
            or target not in frame.columns
        ):
            raise InvalidAnalysisInputError(
                "'target_column' must identify an "
                "existing column."
            )

        restricted_predictors = (
            _validate_predictor_list(
                "restricted_predictors",
                kwargs.get(
                    "restricted_predictors"
                ),
                allow_empty=True,
            )
        )

        full_predictors = (
            _validate_predictor_list(
                "full_predictors",
                kwargs.get(
                    "full_predictors"
                ),
                allow_empty=False,
            )
        )

        if target in full_predictors:
            raise InvalidAnalysisInputError(
                "The target cannot also be a predictor."
            )

        if not set(
            restricted_predictors
        ).issubset(
            set(
                full_predictors
            )
        ):
            raise InvalidAnalysisInputError(
                "The restricted predictors must be "
                "a subset of the full predictors."
            )

        added_predictors = [
            predictor
            for predictor
            in full_predictors
            if predictor
            not in restricted_predictors
        ]

        if not added_predictors:
            raise InvalidAnalysisInputError(
                "The full model must add at least "
                "one predictor."
            )

        missing_columns = [
            column
            for column
            in [
                target,
                *full_predictors,
            ]
            if column
            not in frame.columns
        ]

        if missing_columns:
            raise InvalidAnalysisInputError(
                "Columns were not found: "
                + ", ".join(
                    missing_columns
                )
            )

        fit_intercept = (
            _validate_boolean(
                "fit_intercept",
                kwargs.get(
                    "fit_intercept",
                    True,
                ),
            )
        )

        required = [
            target,
            *full_predictors,
        ]

        numeric = _ensure_numeric(
            frame,
            required,
        )

        complete = numeric[
            required
        ].dropna(
            axis=0,
            how="any",
        )

        sample_size = int(
            len(
                complete
            )
        )

        if sample_size == 0:
            raise InvalidAnalysisInputError(
                "Nested OLS F-testing requires "
                "complete observations."
            )

        restricted = _fit_design(
            complete,
            target,
            restricted_predictors,
            fit_intercept,
        )

        full = _fit_design(
            complete,
            target,
            full_predictors,
            fit_intercept,
        )

        restriction_count = int(
            full[
                "parameter_count"
            ]
            - restricted[
                "parameter_count"
            ]
        )

        if restriction_count <= 0:
            raise InvalidAnalysisInputError(
                "The full model must contain more "
                "parameters than the restricted model."
            )

        rss_restricted = float(
            restricted[
                "rss"
            ]
        )

        rss_full = float(
            full[
                "rss"
            ]
        )

        rss_improvement = float(
            max(
                rss_restricted
                - rss_full,
                0.0,
            )
        )

        denominator_degrees_of_freedom = int(
            full[
                "degrees_of_freedom"
            ]
        )

        if rss_full <= 1e-24:
            if rss_improvement <= 1e-24:
                f_statistic = 0.0
                p_value = 1.0
            else:
                f_statistic = float(
                    "inf"
                )
                p_value = 0.0

        else:
            numerator_mean_square = float(
                rss_improvement
                / restriction_count
            )

            denominator_mean_square = float(
                rss_full
                / denominator_degrees_of_freedom
            )

            f_statistic = float(
                numerator_mean_square
                / denominator_mean_square
            )

            x_value = float(
                (
                    restriction_count
                    * f_statistic
                )
                / (
                    restriction_count
                    * f_statistic
                    + denominator_degrees_of_freedom
                )
            )

            f_cdf = float(
                _regularized_incomplete_beta(
                    x_value,
                    restriction_count
                    / 2.0,
                    denominator_degrees_of_freedom
                    / 2.0,
                )
            )

            p_value = float(
                min(
                    max(
                        1.0
                        - f_cdf,
                        0.0,
                    ),
                    1.0,
                )
            )

        return {
            "statistic_type": (
                "nested_ols_f_test"
            ),
            "target_column": target,
            "restricted_predictors": (
                restricted_predictors
            ),
            "full_predictors": (
                full_predictors
            ),
            "added_predictors": (
                added_predictors
            ),
            "sample_size": sample_size,
            "dropped_count": int(
                len(frame)
                - sample_size
            ),
            "rss_restricted": (
                rss_restricted
            ),
            "rss_full": (
                rss_full
            ),
            "rss_improvement": (
                rss_improvement
            ),
            "restriction_count": (
                restriction_count
            ),
            "numerator_degrees_of_freedom": (
                restriction_count
            ),
            "denominator_degrees_of_freedom": (
                denominator_degrees_of_freedom
            ),
            "f_statistic": (
                None
                if not np.isfinite(
                    f_statistic
                )
                else f_statistic
            ),
            "f_statistic_infinite": bool(
                np.isinf(
                    f_statistic
                )
            ),
            "p_value": (
                p_value
            ),
        }


class JarqueBeraResidualNormalityTest(
    AnalysisMethod
):
    name = "jarque_bera_residual_normality_test"
    category = "regression"
    version = "1.0"

    description = (
        "Assess OLS residual normality with the "
        "Jarque-Bera test based on residual skewness "
        "and excess kurtosis."
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
        fit = _prepare_ols(
            kwargs.get("data"),
            kwargs.get("target_column"),
            kwargs.get("predictor_columns"),
            kwargs.get(
                "fit_intercept",
                True,
            ),
        )

        residuals = fit[
            "residuals"
        ]

        sample_size = int(
            len(
                residuals
            )
        )

        if sample_size < 3:
            raise InvalidAnalysisInputError(
                "Jarque-Bera testing requires at least "
                "three complete residual observations."
            )

        residual_mean = float(
            np.mean(
                residuals
            )
        )

        centered = (
            residuals
            - residual_mean
        )

        second_moment = float(
            np.mean(
                centered ** 2
            )
        )

        scale = float(
            max(
                1.0,
                np.mean(
                    fit[
                        "y"
                    ]
                    ** 2
                ),
            )
        )

        perfect_fit = bool(
            second_moment
            <= (
                np.finfo(float).eps
                * scale
            )
        )

        if perfect_fit:
            skewness = None
            excess_kurtosis = None
            jarque_bera = 0.0
            p_value = 1.0

        else:
            third_moment = float(
                np.mean(
                    centered ** 3
                )
            )

            fourth_moment = float(
                np.mean(
                    centered ** 4
                )
            )

            skewness = float(
                third_moment
                / (
                    second_moment
                    ** 1.5
                )
            )

            excess_kurtosis = float(
                fourth_moment
                / (
                    second_moment
                    ** 2
                )
                - 3.0
            )

            jarque_bera = float(
                sample_size
                / 6.0
                * (
                    skewness ** 2
                    + (
                        excess_kurtosis ** 2
                    )
                    / 4.0
                )
            )

            p_value = (
                1.0
                if jarque_bera == 0.0
                else float(
                    _regularized_gamma_q(
                        1.0,
                        jarque_bera
                        / 2.0,
                    )
                )
            )

        return {
            "diagnostic_type": (
                "jarque_bera_residual_normality_test"
            ),
            "target_column": fit[
                "target"
            ],
            "predictor_columns": fit[
                "predictors"
            ],
            "sample_size": (
                sample_size
            ),
            "dropped_count": fit[
                "dropped_count"
            ],
            "skewness": (
                skewness
            ),
            "excess_kurtosis": (
                excess_kurtosis
            ),
            "jarque_bera_statistic": (
                jarque_bera
            ),
            "degrees_of_freedom": 2,
            "p_value": (
                p_value
            ),
            "perfect_fit": (
                perfect_fit
            ),
            "null_hypothesis": (
                "normally distributed residuals"
            ),
        }
