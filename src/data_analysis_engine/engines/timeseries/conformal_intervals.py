from math import ceil
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
from data_analysis_engine.engines.timeseries.forecast_intervals import (
    _validate_coverage,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _future_timestamps,
    _prepare_regular_series,
    _validate_horizon,
)


def _finite_sample_conformal_radius(
    absolute_errors: np.ndarray,
    *,
    coverage: float,
) -> dict[str, Any]:
    if len(absolute_errors) < 1:
        raise InvalidAnalysisInputError(
            "Conformal calibration requires at least "
            "one absolute forecast error."
        )

    sorted_errors = np.sort(
        np.asarray(
            absolute_errors,
            dtype=float,
        )
    )

    n = len(
        sorted_errors
    )

    rank = int(
        ceil(
            (
                n + 1
            )
            * coverage
        )
    )

    rank = min(
        max(
            rank,
            1,
        ),
        n,
    )

    radius = float(
        sorted_errors[
            rank - 1
        ]
    )

    return {
        "radius": radius,
        "rank": rank,
        "sample_size": n,
        "empirical_quantile_level": float(
            rank / n
        ),
    }


class ConformalForecastInterval(
    AnalysisMethod
):
    name = "conformal_forecast_interval"
    category = "timeseries"
    version = "1.0"

    description = (
        "Construct symmetric finite-sample conformal-style "
        "forecast intervals from rolling-origin absolute "
        "forecast errors."
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

        backtest_kwargs = {
            "time": list(time),
            "values": values.tolist(),
            "model": model,
            "horizon": horizon,
            "stride": stride,
        }

        if (
            kwargs.get(
                "initial_train_size"
            )
            is not None
        ):
            backtest_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        if model == "autoregressive":
            backtest_kwargs[
                "p"
            ] = kwargs.get(
                "p",
                1,
            )

        elif model == "seasonal_naive":
            backtest_kwargs[
                "season_length"
            ] = kwargs.get(
                "season_length",
                7,
            )

        elif (
            model
            == "simple_exponential_smoothing"
        ):
            backtest_kwargs[
                "alpha"
            ] = kwargs.get(
                "alpha",
                0.2,
            )

        elif model == "holt_linear_trend":
            backtest_kwargs[
                "alpha"
            ] = kwargs.get(
                "alpha",
                0.2,
            )

            backtest_kwargs[
                "beta"
            ] = kwargs.get(
                "beta",
                0.1,
            )

        elif model == "holt_winters_additive":
            backtest_kwargs[
                "alpha"
            ] = kwargs.get(
                "alpha",
                0.2,
            )

            backtest_kwargs[
                "beta"
            ] = kwargs.get(
                "beta",
                0.1,
            )

            backtest_kwargs[
                "gamma"
            ] = kwargs.get(
                "gamma",
                0.1,
            )

            backtest_kwargs[
                "season_length"
            ] = kwargs.get(
                "season_length",
                7,
            )

        backtest = (
            RollingOriginForecastBacktest()
            .execute(
                **backtest_kwargs
            )
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

        errors_by_step = {
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
                abs(
                    float(
                        record[
                            "error"
                        ]
                    )
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
            absolute_errors = np.asarray(
                errors_by_step[
                    step
                ],
                dtype=float,
            )

            if (
                len(
                    absolute_errors
                )
                < minimum_calibration_observations
            ):
                raise InvalidAnalysisInputError(
                    "Not enough rolling-origin errors "
                    f"for forecast step {step}. "
                    "Increase the historical sample, "
                    "reduce the horizon, or lower "
                    "'minimum_calibration_observations'."
                )

            calibration = (
                _finite_sample_conformal_radius(
                    absolute_errors,
                    coverage=coverage,
                )
            )

            radius = float(
                calibration[
                    "radius"
                ]
            )

            forecast_value = float(
                point_forecast[
                    step - 1
                ]
            )

            lower_bound = float(
                forecast_value
                - radius
            )

            upper_bound = float(
                forecast_value
                + radius
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
                    "radius": radius,
                    "interval_width": float(
                        2.0
                        * radius
                    ),
                    "calibration_observations": (
                        calibration[
                            "sample_size"
                        ]
                    ),
                    "conformal_rank": (
                        calibration[
                            "rank"
                        ]
                    ),
                    "empirical_quantile_level": (
                        calibration[
                            "empirical_quantile_level"
                        ]
                    ),
                }
            )

        return {
            "interval_type": (
                "symmetric_finite_sample_conformal"
            ),
            "model": model,
            "model_parameters": (
                model_parameters
            ),
            "sample_size": len(
                values
            ),
            "coverage": coverage,
            "horizon": horizon,
            "stride": stride,
            "minimum_calibration_observations": (
                minimum_calibration_observations
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
            "nonconformity_score": (
                "absolute_forecast_error"
            ),
            "finite_sample_correction": True,
            "iid_coverage_guarantee_claimed": False,
            "time_series_assumption_note": (
                "Classical conformal marginal coverage "
                "requires exchangeability. Rolling time-series "
                "forecast errors may be dependent, so the "
                "engine reports a conformal-style interval "
                "without claiming an exact IID guarantee."
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
                for value in point_forecast
            ],
            "intervals": intervals,
        }
