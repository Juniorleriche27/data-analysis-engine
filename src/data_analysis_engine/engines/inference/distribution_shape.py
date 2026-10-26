from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


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
                "Distribution-shape statistics require "
                "genuinely numeric non-missing values."
            )

    values = np.asarray(
        observed,
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise InvalidAnalysisInputError(
            "Distribution-shape statistics require finite "
            "non-missing values. Run "
            "'non_finite_value_diagnostic' first."
        )

    return values


class DistributionShapeStatistics(
    AnalysisMethod
):
    name = "distribution_shape_statistics"
    category = "statistics"
    version = "1.0"

    description = (
        "Calculate moment skewness, excess kurtosis, and "
        "finite-sample bias-corrected shape statistics "
        "for numerical analytical columns."
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
                "Distribution-shape statistics require "
                "at least one observation."
            )

        missing_count = int(
            series.isna().sum()
        )

        observed = series.dropna()

        sample_size = int(
            len(observed)
        )

        if sample_size == 0:
            return {
                "statistic_type": (
                    "distribution_shape_statistics"
                ),
                "total_count": total_count,
                "sample_size": 0,
                "missing_count": missing_count,
                "is_all_missing": True,
                "is_constant": False,
                "mean": None,
                "variance": None,
                "standard_deviation": None,
                "skewness": None,
                "adjusted_skewness": None,
                "excess_kurtosis": None,
                "adjusted_excess_kurtosis": None,
                "kurtosis": None,
                "adjusted_kurtosis": None,
                "shape_status": "all_missing",
            }

        numeric = _numeric_values(
            observed
        )

        mean = float(
            np.mean(numeric)
        )

        centered = (
            numeric
            - mean
        )

        second_moment = float(
            np.mean(
                centered ** 2
            )
        )

        variance = (
            second_moment
        )

        standard_deviation = float(
            np.sqrt(
                variance
            )
        )

        is_constant = bool(
            variance == 0.0
        )

        if is_constant:
            return {
                "statistic_type": (
                    "distribution_shape_statistics"
                ),
                "total_count": total_count,
                "sample_size": sample_size,
                "missing_count": missing_count,
                "is_all_missing": False,
                "is_constant": True,
                "mean": mean,
                "variance": 0.0,
                "standard_deviation": 0.0,
                "skewness": None,
                "adjusted_skewness": None,
                "excess_kurtosis": None,
                "adjusted_excess_kurtosis": None,
                "kurtosis": None,
                "adjusted_kurtosis": None,
                "shape_status": "constant",
            }

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
                second_moment ** 2
            )
            - 3.0
        )

        kurtosis = float(
            excess_kurtosis
            + 3.0
        )

        if sample_size >= 3:
            adjusted_skewness = float(
                np.sqrt(
                    sample_size
                    * (
                        sample_size - 1
                    )
                )
                / (
                    sample_size - 2
                )
                * skewness
            )
        else:
            adjusted_skewness = None

        if sample_size >= 4:
            adjusted_excess_kurtosis = float(
                (
                    sample_size - 1
                )
                / (
                    (
                        sample_size - 2
                    )
                    * (
                        sample_size - 3
                    )
                )
                * (
                    (
                        sample_size + 1
                    )
                    * excess_kurtosis
                    + 6.0
                )
            )

            adjusted_kurtosis = float(
                adjusted_excess_kurtosis
                + 3.0
            )
        else:
            adjusted_excess_kurtosis = None
            adjusted_kurtosis = None

        if skewness > 0.5:
            skew_direction = "right"
        elif skewness < -0.5:
            skew_direction = "left"
        else:
            skew_direction = "approximately_symmetric"

        if excess_kurtosis > 0.5:
            tail_shape = "leptokurtic"
        elif excess_kurtosis < -0.5:
            tail_shape = "platykurtic"
        else:
            tail_shape = "approximately_mesokurtic"

        return {
            "statistic_type": (
                "distribution_shape_statistics"
            ),
            "total_count": total_count,
            "sample_size": sample_size,
            "missing_count": missing_count,
            "is_all_missing": False,
            "is_constant": False,
            "mean": mean,
            "variance": variance,
            "standard_deviation": (
                standard_deviation
            ),
            "skewness": skewness,
            "adjusted_skewness": (
                adjusted_skewness
            ),
            "excess_kurtosis": (
                excess_kurtosis
            ),
            "adjusted_excess_kurtosis": (
                adjusted_excess_kurtosis
            ),
            "kurtosis": kurtosis,
            "adjusted_kurtosis": (
                adjusted_kurtosis
            ),
            "skew_direction": (
                skew_direction
            ),
            "tail_shape": (
                tail_shape
            ),
            "shape_status": "estimated",
            "bias_correction": {
                "skewness_available": bool(
                    sample_size >= 3
                ),
                "kurtosis_available": bool(
                    sample_size >= 4
                ),
            },
        }
