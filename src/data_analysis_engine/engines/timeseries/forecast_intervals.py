from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
    SUPPORTED_MODELS,
    _forecast_from_history,
    _validate_positive_integer,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _future_timestamps,
    _prepare_regular_series,
    _validate_horizon,
)


def _validate_coverage(
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
            "'coverage' must be numerical."
        )

    value = float(value)

    if (
        not np.isfinite(value)
        or value <= 0.0
        or value >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'coverage' must be greater than 0 "
            "and less than 1."
        )

    return value


class EmpiricalForecastInterval(
    AnalysisMethod
):
    name = "empirical_forecast_interval"
    category = "timeseries"
    version = "1.0"

    description = (
        "Construct empirical forecast intervals from "
        "rolling-origin out-of-sample forecast errors."
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

        model = kwargs.get(
            "model",
            "naive",
        )

        if model not in SUPPORTED_MODELS:
            raise InvalidAnalysisInputError(
                "'model' must be one of: "
                + ", ".join(
                    SUPPORTED_MODELS
                )
                + "."
            )

        horizon = _validate_horizon(
            kwargs.get(
                "horizon",
                1,
            )
        )

        coverage = _validate_coverage(
            kwargs.get(
                "coverage",
                0.95,
            )
        )

        stride = _validate_positive_integer(
            kwargs.get(
                "stride",
                1,
            ),
            "stride",
        )

        minimum_interval_observations = (
            _validate_positive_integer(
                kwargs.get(
                    "minimum_interval_observations",
                    3,
                ),
                "minimum_interval_observations",
            )
        )

        backtester = (
            RollingOriginForecastBacktest()
        )

        backtest = backtester.execute(
            time=list(time),
            values=values.tolist(),
            model=model,
            horizon=horizon,
            stride=stride,
            initial_train_size=kwargs.get(
                "initial_train_size"
            ),
            p=kwargs.get(
                "p",
                1,
            ),
            season_length=kwargs.get(
                "season_length",
                7,
            ),
            alpha=kwargs.get(
                "alpha",
                0.2,
            ),
            beta=kwargs.get(
                "beta",
                0.1,
            ),
            gamma=kwargs.get(
                "gamma",
                0.1,
            ),
        )

        model_parameters = (
            backtest[
                "model_parameters"
            ]
        )

        point_forecast = (
            _forecast_from_history(
                values,
                model=model,
                horizon=horizon,
                p=model_parameters.get(
                    "p"
                ),
                season_length=(
                    model_parameters.get(
                        "season_length"
                    )
                ),
                alpha=model_parameters.get(
                    "alpha"
                ),
                beta=model_parameters.get(
                    "beta"
                ),
                gamma=model_parameters.get(
                    "gamma"
                ),
            )
        )

        tail_probability = (
            1.0 - coverage
        )

        lower_quantile = (
            tail_probability
            / 2.0
        )

        upper_quantile = (
            1.0
            - lower_quantile
        )

        errors_by_step: dict[
            int,
            list[float],
        ] = {
            step: []
            for step in range(
                1,
                horizon + 1,
            )
        }

        for record in backtest[
            "records"
        ]:
            step = int(
                record[
                    "forecast_step"
                ]
            )

            errors_by_step[
                step
            ].append(
                float(
                    record[
                        "error"
                    ]
                )
            )

        future_timestamps = (
            _future_timestamps(
                time,
                interval_ns,
                horizon,
            )
        )

        intervals: list[
            dict[str, Any]
        ] = []

        for step in range(
            1,
            horizon + 1,
        ):
            errors = np.asarray(
                errors_by_step[
                    step
                ],
                dtype=float,
            )

            if (
                len(errors)
                < minimum_interval_observations
            ):
                raise InvalidAnalysisInputError(
                    "Not enough rolling-origin errors "
                    f"for forecast step {step}. "
                    "Increase the historical sample, "
                    "reduce the horizon, or lower "
                    "'minimum_interval_observations'."
                )

            lower_error = float(
                np.quantile(
                    errors,
                    lower_quantile,
                )
            )

            upper_error = float(
                np.quantile(
                    errors,
                    upper_quantile,
                )
            )

            forecast_value = float(
                point_forecast[
                    step - 1
                ]
            )

            lower_bound = float(
                forecast_value
                + lower_error
            )

            upper_bound = float(
                forecast_value
                + upper_error
            )

            intervals.append(
                {
                    "forecast_step": step,
                    "timestamp": (
                        future_timestamps[
                            step - 1
                        ]
                    ),
                    "point_forecast": (
                        forecast_value
                    ),
                    "lower_bound": (
                        lower_bound
                    ),
                    "upper_bound": (
                        upper_bound
                    ),
                    "interval_width": float(
                        upper_bound
                        - lower_bound
                    ),
                    "lower_error_quantile": (
                        lower_error
                    ),
                    "upper_error_quantile": (
                        upper_error
                    ),
                    "calibration_observations": (
                        len(errors)
                    ),
                }
            )

        return {
            "interval_type": (
                "empirical_rolling_origin_residual_quantiles"
            ),
            "model": model,
            "model_parameters": (
                model_parameters
            ),
            "sample_size": len(
                values
            ),
            "coverage": (
                coverage
            ),
            "lower_quantile": (
                lower_quantile
            ),
            "upper_quantile": (
                upper_quantile
            ),
            "horizon": (
                horizon
            ),
            "stride": (
                stride
            ),
            "minimum_interval_observations": (
                minimum_interval_observations
            ),
            "initial_train_size": (
                backtest[
                    "initial_train_size"
                ]
            ),
            "calibration_origin_count": (
                backtest[
                    "origin_count"
                ]
            ),
            "look_ahead_leakage": False,
            "error_convention": (
                "actual_minus_forecast"
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "forecast_timestamps": (
                future_timestamps
            ),
            "forecast_values": [
                float(value)
                for value
                in point_forecast
            ],
            "intervals": (
                intervals
            ),
        }
