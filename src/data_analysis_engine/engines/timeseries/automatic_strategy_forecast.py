from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.engines.timeseries.forecast_ensemble import (
    WeightedForecastEnsemble,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    _forecast_from_history,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _future_timestamps,
    _prepare_regular_series,
    _validate_horizon,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    _validate_metric,
    _validate_positive_integer,
)
from data_analysis_engine.engines.timeseries.strategy_selection import (
    ForecastStrategySelection,
)


class AutomaticStrategyForecast(
    AnalysisMethod
):
    name = "automatic_strategy_forecast"
    category = "timeseries"
    version = "1.0"

    description = (
        "Fairly compare individual forecasting models "
        "against an adaptive ensemble, select the best "
        "out-of-sample strategy, and produce the final "
        "forecast using all available historical data."
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

        selection_kwargs = {
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
            "metric": metric,
            "horizon": horizon,
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
            selection_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        selection = (
            ForecastStrategySelection()
            .execute(
                **selection_kwargs
            )
        )

        selected_strategy = (
            selection[
                "best_strategy"
            ]
        )

        selected_strategy_name = (
            selection[
                "best_strategy_name"
            ]
        )

        future_timestamps = (
            _future_timestamps(
                time,
                interval_ns,
                horizon,
            )
        )

        component_forecasts = None
        ensemble_weights = None
        weights_sum = None

        if (
            selected_strategy
            == "single_model"
        ):
            selected_model = selection[
                "best_model"
            ]

            selected_parameters = dict(
                selection[
                    "best_model_parameters"
                ]
                or {}
            )

            forecast_values = (
                _forecast_from_history(
                    values,
                    model=selected_model,
                    horizon=horizon,
                    p=selected_parameters.get(
                        "p"
                    ),
                    season_length=(
                        selected_parameters.get(
                            "season_length"
                        )
                    ),
                    alpha=selected_parameters.get(
                        "alpha"
                    ),
                    beta=selected_parameters.get(
                        "beta"
                    ),
                    gamma=selected_parameters.get(
                        "gamma"
                    ),
                )
            )

            final_strategy_details = {
                "strategy": (
                    "single_model"
                ),
                "model": (
                    selected_model
                ),
                "model_parameters": (
                    selected_parameters
                ),
            }

        else:
            selected_model = None
            selected_parameters = None

            parameters_by_model = {
                candidate[
                    "model"
                ]: dict(
                    candidate[
                        "model_parameters"
                    ]
                    or {}
                )
                for candidate
                in selection[
                    "candidates"
                ]
                if (
                    candidate[
                        "strategy"
                    ]
                    == "single_model"
                )
            }

            ensemble = (
                WeightedForecastEnsemble()
                .execute(
                    time=list(
                        time
                    ),
                    values=(
                        values.tolist()
                    ),
                    models=selection[
                        "models"
                    ],
                    parameters_by_model=(
                        parameters_by_model
                    ),
                    initial_train_size=(
                        selection[
                            "initial_train_size"
                        ]
                    ),
                    horizon=horizon,
                    stride=stride,
                    metric=metric,
                )
            )

            forecast_values = (
                ensemble[
                    "forecast_values"
                ]
            )

            component_forecasts = (
                ensemble[
                    "component_forecasts"
                ]
            )

            ensemble_weights = {
                component[
                    "model"
                ]: component[
                    "weight"
                ]
                for component
                in component_forecasts
            }

            weights_sum = (
                ensemble[
                    "weights_sum"
                ]
            )

            final_strategy_details = {
                "strategy": (
                    "adaptive_ensemble"
                ),
                "models": (
                    selection[
                        "models"
                    ]
                ),
                "weighting_method": (
                    ensemble[
                        "weighting_method"
                    ]
                ),
                "weights": (
                    ensemble_weights
                ),
            }

        forecast_values = [
            float(
                value
            )
            for value
            in forecast_values
        ]

        forecasts = [
            {
                "forecast_step": (
                    step
                ),
                "timestamp": (
                    future_timestamps[
                        step - 1
                    ]
                ),
                "forecast": (
                    forecast_values[
                        step - 1
                    ]
                ),
            }
            for step
            in range(
                1,
                horizon + 1,
            )
        ]

        return {
            "forecast_type": (
                "automatic_fair_strategy_forecast"
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
            "selected_strategy": (
                selected_strategy
            ),
            "selected_strategy_name": (
                selected_strategy_name
            ),
            "selected_model": (
                selected_model
            ),
            "selected_model_parameters": (
                selected_parameters
            ),
            "strategy_selection_score": (
                selection[
                    "best_score"
                ]
            ),
            "strategy_selection_metrics": (
                selection[
                    "best_metrics"
                ]
            ),
            "strategy_ranking": (
                selection[
                    "candidates"
                ]
            ),
            "strategies_evaluated": (
                selection[
                    "strategies_evaluated"
                ]
            ),
            "models_evaluated": (
                selection[
                    "models_evaluated"
                ]
            ),
            "models": (
                selection[
                    "models"
                ]
            ),
            "initial_train_size": (
                selection[
                    "initial_train_size"
                ]
            ),
            "stride": stride,
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
            "common_evaluation_observations": (
                selection[
                    "common_evaluation_observations"
                ]
            ),
            "same_evaluation_window": True,
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "refit_on_full_history": True,
            "ensemble_weights": (
                ensemble_weights
            ),
            "ensemble_weights_sum": (
                weights_sum
            ),
            "component_forecasts": (
                component_forecasts
            ),
            "final_strategy_details": (
                final_strategy_details
            ),
            "forecast_timestamps": (
                future_timestamps
            ),
            "forecast_values": (
                forecast_values
            ),
            "forecasts": forecasts,
        }
