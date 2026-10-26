from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _validate_boolean(
    name: str,
    value: Any,
) -> bool:
    if not isinstance(
        value,
        (bool, np.bool_),
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be boolean."
        )

    return bool(value)


def _validate_columns(
    frame: pd.DataFrame,
    target_column: Any,
    predictor_columns: Any,
) -> tuple[str, list[str]]:
    if (
        not isinstance(
            target_column,
            str,
        )
        or not target_column.strip()
    ):
        raise InvalidAnalysisInputError(
            "'target_column' is required and "
            "must be a non-empty string."
        )

    target = target_column.strip()

    if target not in frame.columns:
        raise InvalidAnalysisInputError(
            f"Target column '{target}' was not found."
        )

    if predictor_columns is None:
        predictors = [
            str(column)
            for column
            in frame.columns
            if str(column) != target
        ]

    else:
        if (
            not isinstance(
                predictor_columns,
                (list, tuple),
            )
            or not predictor_columns
        ):
            raise InvalidAnalysisInputError(
                "'predictor_columns' must contain "
                "at least one column name."
            )

        predictors = [
            str(column)
            for column
            in predictor_columns
        ]

    if len(
        set(predictors)
    ) != len(predictors):
        raise InvalidAnalysisInputError(
            "'predictor_columns' cannot contain duplicates."
        )

    if target in predictors:
        raise InvalidAnalysisInputError(
            "The target column cannot also be a predictor."
        )

    missing = [
        column
        for column
        in predictors
        if column not in frame.columns
    ]

    if missing:
        raise InvalidAnalysisInputError(
            "Predictor columns were not found: "
            + ", ".join(
                missing
            )
        )

    if not predictors:
        raise InvalidAnalysisInputError(
            "OLS regression requires at least "
            "one predictor column."
        )

    return (
        target,
        predictors,
    )


def _ensure_numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    numeric = pd.DataFrame(
        index=frame.index
    )

    for column in columns:
        series = frame[
            column
        ]

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
                    "OLS regression requires genuinely "
                    f"numeric values in column '{column}'."
                )

        try:
            numeric[
                column
            ] = series.astype(
                float
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise InvalidAnalysisInputError(
                "OLS regression requires numeric "
                f"column '{column}'."
            ) from exc

    values = numeric.to_numpy(
        dtype=float
    )

    finite_or_missing = (
        np.isfinite(
            values
        )
        | np.isnan(
            values
        )
    )

    if not np.all(
        finite_or_missing
    ):
        raise InvalidAnalysisInputError(
            "OLS regression requires finite "
            "non-missing values. Run "
            "'non_finite_value_diagnostic' first."
        )

    return numeric


class OrdinaryLeastSquares(
    AnalysisMethod
):
    name = "ordinary_least_squares"
    category = "regression"
    version = "1.0"

    description = (
        "Fit simple or multiple ordinary least squares "
        "regression with deterministic least-squares "
        "estimation and core goodness-of-fit metrics."
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
                "OLS regression requires at least "
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
                "OLS regression requires observations."
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
                "OLS regression requires more complete "
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
                "OLS design matrix is rank deficient. "
                "Remove constant or perfectly collinear "
                "predictors before fitting."
            )

        coefficients, residual_sum_array, _, singular_values = (
            np.linalg.lstsq(
                design,
                y,
                rcond=None,
            )
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

        mean_y = float(
            np.mean(y)
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

        residual_degrees_of_freedom = int(
            sample_size
            - parameter_count
        )

        if (
            fit_intercept
            and sample_size
            > predictor_count + 1
        ):
            adjusted_r_squared = float(
                1.0
                - (
                    1.0 - r_squared
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

        mean_squared_error = float(
            residual_sum_squares
            / sample_size
        )

        rmse = float(
            np.sqrt(
                mean_squared_error
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    residuals
                )
            )
        )

        residual_variance = float(
            residual_sum_squares
            / residual_degrees_of_freedom
        )

        if fit_intercept:
            intercept = float(
                coefficients[0]
            )

            slope_values = coefficients[
                1:
            ]
        else:
            intercept = None
            slope_values = coefficients

        coefficient_map = {
            predictor: float(
                coefficient
            )
            for predictor, coefficient
            in zip(
                predictors,
                slope_values,
            )
        }

        complete_indices = [
            int(index)
            if isinstance(
                index,
                (
                    int,
                    np.integer,
                ),
            )
            else str(index)
            for index
            in complete.index
        ]

        minimum_singular_value = float(
            np.min(
                singular_values
            )
        )

        maximum_singular_value = float(
            np.max(
                singular_values
            )
        )

        condition_number = (
            None
            if minimum_singular_value <= 1e-12
            else float(
                maximum_singular_value
                / minimum_singular_value
            )
        )

        return {
            "model_type": (
                "ordinary_least_squares"
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
            "total_count": (
                total_count
            ),
            "sample_size": (
                sample_size
            ),
            "dropped_count": (
                dropped_count
            ),
            "complete_case_rate": float(
                sample_size
                / total_count
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
            "intercept": (
                intercept
            ),
            "coefficients": (
                coefficient_map
            ),
            "r_squared": (
                r_squared
            ),
            "adjusted_r_squared": (
                adjusted_r_squared
            ),
            "residual_sum_squares": (
                residual_sum_squares
            ),
            "total_sum_squares": (
                total_sum_squares
            ),
            "mean_squared_error": (
                mean_squared_error
            ),
            "root_mean_squared_error": (
                rmse
            ),
            "mean_absolute_error": (
                mae
            ),
            "residual_variance": (
                residual_variance
            ),
            "condition_number": (
                condition_number
            ),
            "complete_indices": (
                complete_indices
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
