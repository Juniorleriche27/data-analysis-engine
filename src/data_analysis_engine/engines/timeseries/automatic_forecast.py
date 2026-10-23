from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
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
from data_analysis_engine.engines.timeseries.model_selection import (
    ForecastModelSelection,
)


class AutomaticForecast(
    AnalysisMethod
):
    name = "automatic_forecast"
    category = "timeseries"
    version = "1.0"

    description = (
        "Automatically compare forecasting models using "
        "rolling-origin out-of-sample evaluation, select "
        "the best model, and produce final forecasts "
        "from the full historical series."
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

        selector = ForecastModelSelection()

        selection_kwargs = {
            "time": list(time),
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

        selection = selector.execute(
            **selection_kwargs
        )

        selected_model = selection[
            "best_model"
        ]

        selected_parameters = dict(
            selection[
                "best_model_parameters"
            ]
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

        forecast_timestamps = (
            _future_timestamps(
                time,
                interval_ns,
                horizon,
            )
        )

        forecasts = [
            {
                "forecast_step": step,
                "timestamp": (
                    forecast_timestamps[
                        step - 1
                    ]
                ),
                "forecast": float(
                    forecast_values[
                        step - 1
                    ]
                ),
            }
            for step in range(
                1,
                horizon + 1,
            )
        ]

        return {
            "forecast_type": (
                "automatic_out_of_sample_model_selection"
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
            "initial_train_size": (
                selection[
                    "initial_train_size"
                ]
            ),
            "stride": stride,
            "models_evaluated": (
                selection[
                    "models_evaluated"
                ]
            ),
            "selected_model": (
                selected_model
            ),
            "selected_model_parameters": (
                selected_parameters
            ),
            "selection_score": float(
                selection[
                    "best_score"
                ]
            ),
            "selection_metrics": (
                selection[
                    "best_metrics"
                ]
            ),
            "model_ranking": (
                selection[
                    "candidates"
                ]
            ),
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "refit_on_full_history": True,
            "forecast_timestamps": (
                forecast_timestamps
            ),
            "forecast_values": [
                float(value)
                for value
                in forecast_values
            ],
            "forecasts": forecasts,
        }
