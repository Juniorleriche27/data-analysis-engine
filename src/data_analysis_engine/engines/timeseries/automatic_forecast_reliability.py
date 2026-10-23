from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.automatic_forecast_intervals import (
    AutomaticForecastWithIntervals,
)
from data_analysis_engine.engines.timeseries.conformal_calibration import (
    ConformalIntervalCalibration,
)
from data_analysis_engine.engines.timeseries.forecast_intervals import (
    _validate_coverage,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _validate_horizon,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    _validate_metric,
    _validate_positive_integer,
)


def _validate_calibration_tolerance(
    value: Any,
) -> float:
    if (
        not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
        or isinstance(
            value,
            (bool, np.bool_),
        )
    ):
        raise InvalidAnalysisInputError(
            "'calibration_tolerance' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value < 0.0
        or value >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'calibration_tolerance' must be greater "
            "than or equal to 0 and less than 1."
        )

    return value


def _calibration_status(
    *,
    empirical_coverage: float,
    nominal_coverage: float,
    tolerance: float,
) -> str:
    gap = (
        empirical_coverage
        - nominal_coverage
    )

    if abs(gap) <= tolerance:
        return "well_calibrated"

    if gap < 0.0:
        return "under_covered"

    return "over_covered"


class AutomaticForecastReliability(
    AnalysisMethod
):
    name = "automatic_forecast_reliability"
    category = "timeseries"
    version = "1.0"

    description = (
        "Automatically select a forecasting model, "
        "produce conformal-style forecast intervals, "
        "and evaluate their historical calibration."
    )

    supported_roles = (
        "time",
        "measure",
    )

    minimum_compatible_columns = 2

    selection_mode = "time_series"
    analysis_unit = "series"

    role_requirements = (
        ("time", 1),
        ("measure", 1),
    )

    allow_constant_columns = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        time, values, interval_ns = (
            _prepare_regular_series(
                kwargs.get("time"),
                kwargs.get("values"),
            )
        )

        horizon = _validate_horizon(
            kwargs.get(
                "horizon",
                1,
            )
        )

        metric = _validate_metric(
            kwargs.get(
                "metric",
                "mae",
            )
        )

        coverage = _validate_coverage(
            kwargs.get(
                "coverage",
                0.90,
            )
        )

        stride = _validate_positive_integer(
            kwargs.get(
                "stride",
                1,
            ),
            "stride",
        )

        minimum_calibration_observations = (
            _validate_positive_integer(
                kwargs.get(
                    "minimum_calibration_observations",
                    3,
                ),
                "minimum_calibration_observations",
            )
        )

        calibration_tolerance = (
            _validate_calibration_tolerance(
                kwargs.get(
                    "calibration_tolerance",
                    0.10,
                )
            )
        )

        forecast_kwargs = {
            "time": list(time),
            "values": values.tolist(),
            "models": kwargs.get(
                "models"
            ),
            "parameters_by_model": kwargs.get(
                "parameters_by_model"
            ),
            "horizon": horizon,
            "metric": metric,
            "coverage": coverage,
            "stride": stride,
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
        }

        if (
            kwargs.get(
                "initial_train_size"
            )
            is not None
        ):
            forecast_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        forecast = (
            AutomaticForecastWithIntervals()
            .execute(
                **forecast_kwargs
            )
        )

        selected_model = forecast[
            "selected_model"
        ]

        selected_parameters = dict(
            forecast[
                "selected_model_parameters"
            ]
        )

        calibration_kwargs = {
            "time": list(time),
            "values": values.tolist(),
            "model": selected_model,
            "initial_train_size": (
                forecast[
                    "initial_train_size"
                ]
            ),
            "horizon": horizon,
            "coverage": coverage,
            "stride": stride,
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
            **selected_parameters,
        }

        calibration = (
            ConformalIntervalCalibration()
            .execute(
                **calibration_kwargs
            )
        )

        empirical_coverage = float(
            calibration[
                "empirical_coverage"
            ]
        )

        coverage_gap = float(
            calibration[
                "coverage_gap"
            ]
        )

        status = _calibration_status(
            empirical_coverage=(
                empirical_coverage
            ),
            nominal_coverage=coverage,
            tolerance=calibration_tolerance,
        )

        reliable_within_tolerance = (
            status
            == "well_calibrated"
        )

        return {
            "analysis_type": (
                "automatic_forecast_reliability"
            ),
            "sample_size": len(
                values
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "horizon": horizon,
            "selection_metric": (
                metric
            ),
            "coverage": coverage,
            "calibration_tolerance": (
                calibration_tolerance
            ),
            "selected_model": (
                selected_model
            ),
            "selected_model_parameters": (
                selected_parameters
            ),
            "selection_score": (
                forecast[
                    "selection_score"
                ]
            ),
            "selection_metrics": (
                forecast[
                    "selection_metrics"
                ]
            ),
            "models_evaluated": (
                forecast[
                    "models_evaluated"
                ]
            ),
            "model_ranking": (
                forecast[
                    "model_ranking"
                ]
            ),
            "initial_train_size": (
                forecast[
                    "initial_train_size"
                ]
            ),
            "stride": stride,
            "forecast_values": (
                forecast[
                    "forecast_values"
                ]
            ),
            "forecast_timestamps": (
                forecast[
                    "forecast_timestamps"
                ]
            ),
            "intervals": (
                forecast[
                    "intervals"
                ]
            ),
            "forecasts": (
                forecast[
                    "forecasts"
                ]
            ),
            "nominal_coverage": (
                coverage
            ),
            "empirical_coverage": (
                empirical_coverage
            ),
            "coverage_gap": (
                coverage_gap
            ),
            "absolute_coverage_error": (
                calibration[
                    "absolute_coverage_error"
                ]
            ),
            "calibration_status": (
                status
            ),
            "reliable_within_tolerance": (
                reliable_within_tolerance
            ),
            "validation_observations": (
                calibration[
                    "validation_observations"
                ]
            ),
            "covered_observations": (
                calibration[
                    "covered_observations"
                ]
            ),
            "average_interval_width": (
                calibration[
                    "average_interval_width"
                ]
            ),
            "median_interval_width": (
                calibration[
                    "median_interval_width"
                ]
            ),
            "per_horizon_calibration": (
                calibration[
                    "per_horizon"
                ]
            ),
            "look_ahead_leakage": False,
            "finite_sample_correction": True,
            "refit_on_full_history": True,
            "iid_coverage_guarantee_claimed": False,
            "post_selection_diagnostic": True,
            "selection_adjusted_coverage_guarantee": False,
            "diagnostic_note": (
                "Historical calibration is evaluated "
                "after model selection. It is a useful "
                "diagnostic, but it is not an independent "
                "selection-adjusted coverage guarantee."
            ),
        }
