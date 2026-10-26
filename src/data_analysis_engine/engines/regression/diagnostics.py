from numbers import Real
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
from data_analysis_engine.engines.regression.ols import (
    _ensure_numeric,
    _validate_boolean,
    _validate_columns,
)


def _validate_significance_level(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'significance_level' must be numeric."
        )

    significance_level = float(value)

    if (
        not np.isfinite(significance_level)
        or significance_level <= 0.0
        or significance_level >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'significance_level' must satisfy "
            "0 < significance_level < 1."
        )

    return significance_level


def _prepare_ols(
    data: Any,
    target_column: Any,
    predictor_columns: Any,
    fit_intercept: Any,
) -> dict[str, Any]:
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
            "OLS diagnostics require at least "
            "two numerical columns."
        )

    try:
        frame = pd.DataFrame(
            data
        )
    except Exception as exc:
        raise InvalidAnalysisInputError(
            "All columns must form a rectangular dataframe."
        ) from exc

    total_count = int(
        len(frame)
    )

    if total_count == 0:
        raise InvalidAnalysisInputError(
            "OLS diagnostics require observations."
        )

    target, predictors = _validate_columns(
        frame,
        target_column,
        predictor_columns,
    )

    fit_intercept = _validate_boolean(
        "fit_intercept",
        fit_intercept,
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
            "OLS diagnostics require more complete "
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
    else:
        design = x

    matrix_rank = int(
        np.linalg.matrix_rank(
            design
        )
    )

    if matrix_rank < parameter_count:
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

    residual_sum_squares = float(
        np.sum(
            residuals ** 2
        )
    )

    residual_degrees_of_freedom = int(
        sample_size
        - parameter_count
    )

    residual_variance = float(
        residual_sum_squares
        / residual_degrees_of_freedom
    )

    return {
        "frame": frame,
        "complete": complete,
        "target": target,
        "predictors": predictors,
        "total_count": total_count,
        "sample_size": sample_size,
        "dropped_count": int(
            total_count
            - sample_size
        ),
        "predictor_count": predictor_count,
        "parameter_count": parameter_count,
        "fit_intercept": fit_intercept,
        "x": x,
        "y": y,
        "design": design,
        "coefficients": coefficients,
        "predictions": predictions,
        "residuals": residuals,
        "residual_sum_squares": (
            residual_sum_squares
        ),
        "residual_degrees_of_freedom": (
            residual_degrees_of_freedom
        ),
        "residual_variance": (
            residual_variance
        ),
    }


class BreuschPaganHeteroskedasticityTest(
    AnalysisMethod
):
    name = "breusch_pagan_heteroskedasticity_test"
    category = "regression"
    version = "1.0"

    description = (
        "Diagnose heteroskedasticity in OLS residuals "
        "using the Breusch-Pagan Lagrange multiplier test."
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
        significance_level = (
            _validate_significance_level(
                kwargs.get(
                    "significance_level",
                    0.05,
                )
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

        x = fit[
            "x"
        ]

        residuals = fit[
            "residuals"
        ]

        sample_size = fit[
            "sample_size"
        ]

        predictor_count = fit[
            "predictor_count"
        ]

        auxiliary_design = np.column_stack(
            [
                np.ones(
                    sample_size,
                    dtype=float,
                ),
                x,
            ]
        )

        auxiliary_parameter_count = int(
            predictor_count + 1
        )

        if np.linalg.matrix_rank(
            auxiliary_design
        ) < auxiliary_parameter_count:
            raise InvalidAnalysisInputError(
                "Breusch-Pagan auxiliary regression "
                "is rank deficient."
            )

        squared_residuals = (
            residuals ** 2
        )

        auxiliary_coefficients, *_ = np.linalg.lstsq(
            auxiliary_design,
            squared_residuals,
            rcond=None,
        )

        auxiliary_predictions = (
            auxiliary_design
            @ auxiliary_coefficients
        )

        auxiliary_total_sum_squares = float(
            np.sum(
                (
                    squared_residuals
                    - np.mean(
                        squared_residuals
                    )
                )
                ** 2
            )
        )

        auxiliary_residual_sum_squares = float(
            np.sum(
                (
                    squared_residuals
                    - auxiliary_predictions
                )
                ** 2
            )
        )

        if auxiliary_total_sum_squares <= 1e-24:
            auxiliary_r_squared = 0.0
        else:
            auxiliary_r_squared = float(
                1.0
                - auxiliary_residual_sum_squares
                / auxiliary_total_sum_squares
            )

            auxiliary_r_squared = float(
                min(
                    max(
                        auxiliary_r_squared,
                        0.0,
                    ),
                    1.0,
                )
            )

        lm_statistic = float(
            sample_size
            * auxiliary_r_squared
        )

        degrees_of_freedom = int(
            predictor_count
        )

        p_value = (
            1.0
            if lm_statistic == 0.0
            else float(
                _regularized_gamma_q(
                    degrees_of_freedom
                    / 2.0,
                    lm_statistic
                    / 2.0,
                )
            )
        )

        detected = bool(
            p_value
            < significance_level
        )

        return {
            "diagnostic_type": (
                "breusch_pagan_heteroskedasticity_test"
            ),
            "target_column": fit[
                "target"
            ],
            "predictor_columns": fit[
                "predictors"
            ],
            "sample_size": sample_size,
            "dropped_count": fit[
                "dropped_count"
            ],
            "significance_level": (
                significance_level
            ),
            "lm_statistic": (
                lm_statistic
            ),
            "degrees_of_freedom": (
                degrees_of_freedom
            ),
            "p_value": (
                p_value
            ),
            "auxiliary_r_squared": (
                auxiliary_r_squared
            ),
            "heteroskedasticity_detected": (
                detected
            ),
            "null_hypothesis": (
                "constant residual variance"
            ),
        }


class OLSResidualAutocorrelationDiagnostic(
    AnalysisMethod
):
    name = "ols_residual_autocorrelation_diagnostic"
    category = "regression"
    version = "1.0"

    description = (
        "Diagnose ordered OLS residual dependence using "
        "the Durbin-Watson statistic and lag-1 residual "
        "autocorrelation."
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

        if len(
            residuals
        ) < 3:
            raise InvalidAnalysisInputError(
                "Residual autocorrelation diagnostics "
                "require at least three complete observations."
            )

        denominator = float(
            np.sum(
                residuals ** 2
            )
        )

        scale = float(
            max(
                1.0,
                np.sum(
                    fit[
                        "y"
                    ]
                    ** 2
                ),
            )
        )

        perfect_fit = bool(
            denominator
            <= (
                np.finfo(float).eps
                * scale
            )
        )

        if perfect_fit:
            durbin_watson = None
            lag1_autocorrelation = None
            interpretation = "perfect_fit"

        else:
            numerator = float(
                np.sum(
                    np.diff(
                        residuals
                    )
                    ** 2
                )
            )

            durbin_watson = float(
                numerator
                / denominator
            )

            previous = residuals[
                :-1
            ]

            current = residuals[
                1:
            ]

            if (
                np.std(previous) == 0.0
                or np.std(current) == 0.0
            ):
                lag1_autocorrelation = None
            else:
                lag1_autocorrelation = float(
                    np.corrcoef(
                        previous,
                        current,
                    )[0, 1]
                )

            if durbin_watson < 1.5:
                interpretation = (
                    "possible_positive_autocorrelation"
                )
            elif durbin_watson > 2.5:
                interpretation = (
                    "possible_negative_autocorrelation"
                )
            else:
                interpretation = (
                    "limited_autocorrelation_signal"
                )

        return {
            "diagnostic_type": (
                "ols_residual_autocorrelation_diagnostic"
            ),
            "target_column": fit[
                "target"
            ],
            "predictor_columns": fit[
                "predictors"
            ],
            "sample_size": fit[
                "sample_size"
            ],
            "dropped_count": fit[
                "dropped_count"
            ],
            "durbin_watson": (
                durbin_watson
            ),
            "lag1_residual_autocorrelation": (
                lag1_autocorrelation
            ),
            "perfect_fit": (
                perfect_fit
            ),
            "interpretation": (
                interpretation
            ),
            "order_sensitive": True,
        }


class OLSInfluenceDiagnostic(
    AnalysisMethod
):
    name = "ols_influence_diagnostic"
    category = "regression"
    version = "1.0"

    description = (
        "Measure OLS observation leverage, standardized "
        "residuals, and Cook's distance to identify "
        "potentially influential observations."
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

        design = fit[
            "design"
        ]

        residuals = fit[
            "residuals"
        ]

        sample_size = fit[
            "sample_size"
        ]

        parameter_count = fit[
            "parameter_count"
        ]

        xtx_inverse = np.linalg.inv(
            design.T
            @ design
        )

        leverage = np.einsum(
            "ij,jk,ik->i",
            design,
            xtx_inverse,
            design,
        )

        leverage = np.clip(
            leverage,
            0.0,
            1.0,
        )

        mean_leverage = float(
            parameter_count
            / sample_size
        )

        high_leverage_threshold = float(
            2.0
            * parameter_count
            / sample_size
        )

        cooks_distance_threshold = float(
            4.0
            / sample_size
        )

        residual_variance = fit[
            "residual_variance"
        ]

        standardized_residuals: list[
            float | None
        ] = []

        cooks_distances: list[
            float | None
        ] = []

        cooks_infinite: list[
            bool
        ] = []

        for residual, hat_value in zip(
            residuals,
            leverage,
        ):
            one_minus_hat = float(
                1.0
                - hat_value
            )

            if residual_variance <= 1e-24:
                standardized = 0.0
                cooks_distance = 0.0
                is_infinite = False

            else:
                denominator = float(
                    np.sqrt(
                        residual_variance
                        * max(
                            one_minus_hat,
                            0.0,
                        )
                    )
                )

                if denominator <= 1e-15:
                    standardized = None
                else:
                    standardized = float(
                        residual
                        / denominator
                    )

                if one_minus_hat <= 1e-15:
                    if abs(
                        residual
                    ) <= 1e-15:
                        cooks_distance = 0.0
                        is_infinite = False
                    else:
                        cooks_distance = None
                        is_infinite = True
                else:
                    cooks_distance = float(
                        (
                            residual ** 2
                        )
                        / (
                            parameter_count
                            * residual_variance
                        )
                        * (
                            hat_value
                            / (
                                one_minus_hat ** 2
                            )
                        )
                    )

                    is_infinite = False

            standardized_residuals.append(
                standardized
            )

            cooks_distances.append(
                cooks_distance
            )

            cooks_infinite.append(
                is_infinite
            )

        original_indices = list(
            fit[
                "complete"
            ].index
        )

        serialized_indices = [
            (
                int(index)
                if isinstance(
                    index,
                    (
                        int,
                        np.integer,
                    ),
                )
                else str(index)
            )
            for index
            in original_indices
        ]

        high_leverage_positions = [
            serialized_indices[index]
            for index, value
            in enumerate(
                leverage
            )
            if value
            > high_leverage_threshold
        ]

        influential_positions = []

        for index, (
            cooks_distance,
            infinite,
        ) in enumerate(
            zip(
                cooks_distances,
                cooks_infinite,
            )
        ):
            if (
                infinite
                or (
                    cooks_distance is not None
                    and cooks_distance
                    > cooks_distance_threshold
                )
            ):
                influential_positions.append(
                    serialized_indices[
                        index
                    ]
                )

        return {
            "diagnostic_type": (
                "ols_influence_diagnostic"
            ),
            "target_column": fit[
                "target"
            ],
            "predictor_columns": fit[
                "predictors"
            ],
            "sample_size": sample_size,
            "parameter_count": (
                parameter_count
            ),
            "dropped_count": fit[
                "dropped_count"
            ],
            "mean_leverage": (
                mean_leverage
            ),
            "high_leverage_threshold": (
                high_leverage_threshold
            ),
            "cooks_distance_threshold": (
                cooks_distance_threshold
            ),
            "observation_indices": (
                serialized_indices
            ),
            "leverage": [
                float(value)
                for value
                in leverage
            ],
            "standardized_residuals": (
                standardized_residuals
            ),
            "cooks_distance": (
                cooks_distances
            ),
            "cooks_distance_infinite": (
                cooks_infinite
            ),
            "high_leverage_positions": (
                high_leverage_positions
            ),
            "influential_positions": (
                influential_positions
            ),
            "high_leverage_count": int(
                len(
                    high_leverage_positions
                )
            ),
            "influential_count": int(
                len(
                    influential_positions
                )
            ),
        }
