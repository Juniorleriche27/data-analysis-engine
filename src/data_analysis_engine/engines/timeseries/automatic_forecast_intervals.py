from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.engines.timeseries.automatic_forecast import (
    AutomaticForecast,
)
from data_analysis_engine.engines.timeseries.conformal_intervals import (
    ConformalForecastInterval,
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


class AutomaticForecastWithIntervals(
    AnalysisMethod
):
    name = "automatic_forecast_with_intervals"
    category = "timeseries"
    version = "1.0"

    description = (
        "Automatically select the best forecasting model "
        "using rolling-origin evaluation, forecast from the "
        "full history, and construct finite-sample "
        "conformal-style forecast intervals."
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

        automatic_kwargs = {
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
            "stride": stride,
        }

        if (
            kwargs.get(
                "initial_train_size"
            )
            is not None
        ):
            automatic_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        automatic = (
            AutomaticForecast()
            .execute(
                **automatic_kwargs
            )
        )

        selected_model = automatic[
            "selected_model"
        ]

        selected_parameters = dict(
            automatic[
                "selected_model_parameters"
            ]
        )

        interval_kwargs = {
            "time": list(time),
            "values": values.tolist(),
            "model": selected_model,
            "initial_train_size": (
                automatic[
                    "initial_train_size"
                ]
            ),
            "horizon": horizon,
            "stride": stride,
            "coverage": coverage,
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
            **selected_parameters,
        }

        conformal = (
            ConformalForecastInterval()
            .execute(
                **interval_kwargs
            )
        )

        forecasts = []

        for interval in conformal[
            "intervals"
        ]:
            forecasts.append(
                {
                    "forecast_step": (
                        interval[
                            "forecast_step"
                        ]
                    ),
                    "timestamp": (
                        interval[
                            "timestamp"
                        ]
                    ),
                    "forecast": (
                        interval[
                            "point_forecast"
                        ]
                    ),
                    "lower_bound": (
                        interval[
                            "lower_bound"
                        ]
                    ),
                    "upper_bound": (
                        interval[
                            "upper_bound"
                        ]
                    ),
                    "radius": (
                        interval[
                            "radius"
                        ]
                    ),
                    "interval_width": (
                        interval[
                            "interval_width"
                        ]
                    ),
                    "calibration_observations": (
                        interval[
                            "calibration_observations"
                        ]
                    ),
                    "conformal_rank": (
                        interval[
                            "conformal_rank"
                        ]
                    ),
                }
            )

        return {
            "forecast_type": (
                "automatic_model_selection_with_"
                "conformal_intervals"
            ),
            "interval_type": (
                conformal[
                    "interval_type"
                ]
            ),
            "sample_size": len(
                values
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "horizon": horizon,
            "coverage": coverage,
            "selection_metric": (
                metric
            ),
            "selected_model": (
                selected_model
            ),
            "selected_model_parameters": (
                selected_parameters
            ),
            "selection_score": (
                automatic[
                    "selection_score"
                ]
            ),
            "selection_metrics": (
                automatic[
                    "selection_metrics"
                ]
            ),
            "model_ranking": (
                automatic[
                    "model_ranking"
                ]
            ),
            "models_evaluated": (
                automatic[
                    "models_evaluated"
                ]
            ),
            "initial_train_size": (
                automatic[
                    "initial_train_size"
                ]
            ),
            "stride": stride,
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
            "calibration_origin_count": (
                conformal[
                    "calibration_origin_count"
                ]
            ),
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "refit_on_full_history": True,
            "finite_sample_correction": True,
            "nonconformity_score": (
                "absolute_forecast_error"
            ),
            "iid_coverage_guarantee_claimed": False,
            "forecast_timestamps": (
                conformal[
                    "forecast_timestamps"
                ]
            ),
            "forecast_values": (
                conformal[
                    "forecast_values"
                ]
            ),
            "intervals": (
                conformal[
                    "intervals"
                ]
            ),
            "forecasts": forecasts,
        }
