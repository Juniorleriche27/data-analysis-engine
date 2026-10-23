from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.automatic_strategy_forecast import (
    AutomaticStrategyForecast,
)
from data_analysis_engine.engines.timeseries.conformal_intervals import (
    _finite_sample_conformal_radius,
)
from data_analysis_engine.engines.timeseries.ensemble_backtest import (
    AdaptiveForecastEnsembleBacktest,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
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


class AutomaticStrategyForecastWithIntervals(
    AnalysisMethod
):
    name = (
        "automatic_strategy_forecast_with_intervals"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Select the best forecasting strategy between "
        "individual models and an adaptive ensemble, "
        "produce the final forecast, and construct "
        "finite-sample conformal-style intervals from "
        "historical errors of the selected strategy."
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

        minimum_weight_observations = (
            _validate_positive_integer(
                kwargs.get(
                    "minimum_weight_observations",
                    3,
                ),
                "minimum_weight_observations",
            )
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
            "time": list(
                time
            ),
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
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
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
            AutomaticStrategyForecast()
            .execute(
                **automatic_kwargs
            )
        )

        selected_strategy = (
            automatic[
                "selected_strategy"
            ]
        )

        selected_model = (
            automatic[
                "selected_model"
            ]
        )

        selected_parameters = (
            automatic[
                "selected_model_parameters"
            ]
        )

        if (
            selected_strategy
            == "single_model"
        ):
            selected_parameters = dict(
                selected_parameters
                or {}
            )

            calibration = (
                RollingOriginForecastBacktest()
                .execute(
                    time=list(
                        time
                    ),
                    values=(
                        values.tolist()
                    ),
                    model=selected_model,
                    initial_train_size=(
                        automatic[
                            "initial_train_size"
                        ]
                    ),
                    horizon=horizon,
                    stride=stride,
                    **selected_parameters,
                )
            )

            calibration_records = list(
                calibration[
                    "records"
                ]
            )

            calibration_source = (
                "selected_single_model_backtest"
            )

            calibration_origin_count = (
                calibration[
                    "origin_count"
                ]
            )

        elif (
            selected_strategy
            == "adaptive_ensemble"
        ):
            calibration = (
                AdaptiveForecastEnsembleBacktest()
                .execute(
                    time=list(
                        time
                    ),
                    values=(
                        values.tolist()
                    ),
                    models=automatic[
                        "models"
                    ],
                    parameters_by_model=(
                        kwargs.get(
                            "parameters_by_model"
                        )
                    ),
                    initial_train_size=(
                        automatic[
                            "initial_train_size"
                        ]
                    ),
                    horizon=horizon,
                    stride=stride,
                    metric=metric,
                    minimum_weight_observations=(
                        minimum_weight_observations
                    ),
                )
            )

            calibration_records = list(
                calibration[
                    "records"
                ]
            )

            calibration_source = (
                "adaptive_ensemble_backtest"
            )

            calibration_origin_count = len(
                {
                    int(
                        record[
                            "origin"
                        ]
                    )
                    for record
                    in calibration_records
                }
            )

        else:
            raise InvalidAnalysisInputError(
                "Unsupported selected forecast strategy: "
                f"'{selected_strategy}'."
            )

        absolute_errors_by_step = {
            step: []
            for step
            in range(
                1,
                horizon + 1,
            )
        }

        for record in calibration_records:
            step = int(
                record[
                    "forecast_step"
                ]
            )

            absolute_errors_by_step[
                step
            ].append(
                abs(
                    float(
                        record[
                            "error"
                        ]
                    )
                )
            )

        forecast_values = [
            float(
                value
            )
            for value
            in automatic[
                "forecast_values"
            ]
        ]

        forecast_timestamps = (
            automatic[
                "forecast_timestamps"
            ]
        )

        intervals = []

        for step in range(
            1,
            horizon + 1,
        ):
            errors = np.asarray(
                absolute_errors_by_step[
                    step
                ],
                dtype=float,
            )

            if (
                len(errors)
                < minimum_calibration_observations
            ):
                raise InvalidAnalysisInputError(
                    "Not enough selected-strategy "
                    "forecast errors for forecast "
                    f"step {step}. Increase the "
                    "historical sample, reduce the "
                    "horizon, or lower "
                    "'minimum_calibration_observations'."
                )

            conformal = (
                _finite_sample_conformal_radius(
                    errors,
                    coverage=coverage,
                )
            )

            radius = float(
                conformal[
                    "radius"
                ]
            )

            point_forecast = (
                forecast_values[
                    step - 1
                ]
            )

            lower_bound = float(
                point_forecast
                - radius
            )

            upper_bound = float(
                point_forecast
                + radius
            )

            intervals.append(
                {
                    "forecast_step": (
                        step
                    ),
                    "timestamp": (
                        forecast_timestamps[
                            step - 1
                        ]
                    ),
                    "point_forecast": (
                        point_forecast
                    ),
                    "lower_bound": (
                        lower_bound
                    ),
                    "upper_bound": (
                        upper_bound
                    ),
                    "radius": radius,
                    "interval_width": float(
                        2.0
                        * radius
                    ),
                    "calibration_observations": (
                        conformal[
                            "sample_size"
                        ]
                    ),
                    "conformal_rank": (
                        conformal[
                            "rank"
                        ]
                    ),
                    "empirical_quantile_level": (
                        conformal[
                            "empirical_quantile_level"
                        ]
                    ),
                }
            )

        forecasts = [
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
            }
            for interval
            in intervals
        ]

        return {
            "forecast_type": (
                "automatic_strategy_forecast_"
                "with_conformal_intervals"
            ),
            "interval_type": (
                "symmetric_finite_sample_conformal"
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
            "selected_strategy": (
                selected_strategy
            ),
            "selected_strategy_name": (
                automatic[
                    "selected_strategy_name"
                ]
            ),
            "selected_model": (
                selected_model
            ),
            "selected_model_parameters": (
                selected_parameters
            ),
            "strategy_selection_score": (
                automatic[
                    "strategy_selection_score"
                ]
            ),
            "strategy_selection_metrics": (
                automatic[
                    "strategy_selection_metrics"
                ]
            ),
            "strategy_ranking": (
                automatic[
                    "strategy_ranking"
                ]
            ),
            "strategies_evaluated": (
                automatic[
                    "strategies_evaluated"
                ]
            ),
            "models_evaluated": (
                automatic[
                    "models_evaluated"
                ]
            ),
            "models": (
                automatic[
                    "models"
                ]
            ),
            "initial_train_size": (
                automatic[
                    "initial_train_size"
                ]
            ),
            "stride": stride,
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
            "calibration_source": (
                calibration_source
            ),
            "calibration_origin_count": (
                calibration_origin_count
            ),
            "common_evaluation_observations": (
                automatic[
                    "common_evaluation_observations"
                ]
            ),
            "same_evaluation_window": True,
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "refit_on_full_history": True,
            "finite_sample_correction": True,
            "nonconformity_score": (
                "absolute_forecast_error"
            ),
            "iid_coverage_guarantee_claimed": False,
            "selection_adjusted_coverage_guarantee": False,
            "post_selection_interval": True,
            "diagnostic_note": (
                "Intervals are calibrated from historical "
                "errors of the selected strategy. Because "
                "the strategy itself was selected from the "
                "same historical sample, the engine does "
                "not claim an independent selection-adjusted "
                "coverage guarantee."
            ),
            "ensemble_weights": (
                automatic[
                    "ensemble_weights"
                ]
            ),
            "forecast_timestamps": (
                forecast_timestamps
            ),
            "forecast_values": (
                forecast_values
            ),
            "intervals": intervals,
            "forecasts": forecasts,
        }
