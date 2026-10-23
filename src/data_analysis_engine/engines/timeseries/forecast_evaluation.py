from math import sqrt
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.autoregression import (
    _fit_autoregression,
    _recursive_ar_forecast,
    _validate_order,
)
from data_analysis_engine.engines.timeseries.exponential_smoothing import (
    _simple_exponential_smoothing_forecast,
    _validate_alpha,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _recursive_seasonal_naive_forecast,
    _validate_season_length,
)
from data_analysis_engine.engines.timeseries.holt import (
    _holt_linear_forecast,
    _validate_beta,
)
from data_analysis_engine.engines.timeseries.holt_winters import (
    _holt_winters_additive_forecast,
    _validate_gamma,
)


SUPPORTED_BASELINES = (
    "naive",
    "drift",
    "seasonal_naive",
)

SUPPORTED_MODELS = (
    "naive",
    "drift",
    "seasonal_naive",
    "autoregressive",
    "simple_exponential_smoothing",
    "holt_linear_trend",
    "holt_winters_additive",
)


def _validate_positive_integer(
    value: Any,
    name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be an integer."
        )

    if value < 1:
        raise InvalidAnalysisInputError(
            f"'{name}' must be at least 1."
        )

    return value


def _forecast_from_history(
    history: np.ndarray,
    *,
    model: str,
    horizon: int,
    p: int | None = None,
    season_length: int | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    if model == "naive":
        return np.full(
            horizon,
            float(history[-1]),
            dtype=float,
        )

    if model == "drift":
        if len(history) < 2:
            raise InvalidAnalysisInputError(
                "Drift forecasting requires at least "
                "two training observations."
            )

        drift = float(
            (
                history[-1]
                - history[0]
            )
            / (
                len(history)
                - 1
            )
        )

        return np.asarray(
            [
                float(
                    history[-1]
                    + drift * step
                )
                for step in range(
                    1,
                    horizon + 1,
                )
            ],
            dtype=float,
        )

    if model == "seasonal_naive":
        if season_length is None:
            raise InvalidAnalysisInputError(
                "Seasonal naive forecasting requires "
                "'season_length'."
            )

        return _recursive_seasonal_naive_forecast(
            history,
            season_length=season_length,
            horizon=horizon,
        )

    if model == "autoregressive":
        if p is None:
            raise InvalidAnalysisInputError(
                "Autoregressive forecasting requires 'p'."
            )

        fit = _fit_autoregression(
            history,
            p,
        )

        return _recursive_ar_forecast(
            history,
            intercept=fit[
                "intercept"
            ],
            lag_coefficients=fit[
                "lag_coefficients"
            ],
            horizon=horizon,
        )

    if (
        model
        == "simple_exponential_smoothing"
    ):
        if alpha is None:
            raise InvalidAnalysisInputError(
                "Simple exponential smoothing "
                "requires 'alpha'."
            )

        return (
            _simple_exponential_smoothing_forecast(
                history,
                alpha=alpha,
                horizon=horizon,
            )
        )

    if model == "holt_linear_trend":
        if alpha is None:
            raise InvalidAnalysisInputError(
                "Holt linear trend requires 'alpha'."
            )

        if beta is None:
            raise InvalidAnalysisInputError(
                "Holt linear trend requires 'beta'."
            )

        return _holt_linear_forecast(
            history,
            alpha=alpha,
            beta=beta,
            horizon=horizon,
        )

    if model == "holt_winters_additive":
        if alpha is None:
            raise InvalidAnalysisInputError(
                "Additive Holt-Winters requires 'alpha'."
            )

        if beta is None:
            raise InvalidAnalysisInputError(
                "Additive Holt-Winters requires 'beta'."
            )

        if gamma is None:
            raise InvalidAnalysisInputError(
                "Additive Holt-Winters requires 'gamma'."
            )

        if season_length is None:
            raise InvalidAnalysisInputError(
                "Additive Holt-Winters requires "
                "'season_length'."
            )

        return _holt_winters_additive_forecast(
            history,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            season_length=season_length,
            horizon=horizon,
        )

    raise InvalidAnalysisInputError(
        "Unsupported forecast model: "
        f"'{model}'."
    )


def _forecast_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict[str, Any]:
    errors = (
        actual
        - forecast
    )

    absolute_errors = np.abs(
        errors
    )

    squared_errors = (
        errors ** 2
    )

    mae = float(
        np.mean(
            absolute_errors
        )
    )

    rmse = float(
        sqrt(
            float(
                np.mean(
                    squared_errors
                )
            )
        )
    )

    nonzero_actual = (
        actual != 0
    )

    if np.any(
        nonzero_actual
    ):
        mape = float(
            np.mean(
                absolute_errors[
                    nonzero_actual
                ]
                / np.abs(
                    actual[
                        nonzero_actual
                    ]
                )
            )
            * 100.0
        )

        mape_defined = True

        mape_observations = int(
            np.sum(
                nonzero_actual
            )
        )

    else:
        mape = None
        mape_defined = False
        mape_observations = 0

    smape_denominator = (
        np.abs(actual)
        + np.abs(forecast)
    )

    smape_terms = np.zeros(
        len(actual),
        dtype=float,
    )

    nonzero_denominator = (
        smape_denominator > 0
    )

    smape_terms[
        nonzero_denominator
    ] = (
        2.0
        * absolute_errors[
            nonzero_denominator
        ]
        / smape_denominator[
            nonzero_denominator
        ]
    )

    smape = float(
        np.mean(
            smape_terms
        )
        * 100.0
    )

    mean_error = float(
        np.mean(
            errors
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mape_defined": mape_defined,
        "mape_observations": (
            mape_observations
        ),
        "smape": smape,
        "mean_error": mean_error,
        "error_convention": (
            "actual_minus_forecast"
        ),
    }


class RollingOriginForecastBacktest(
    AnalysisMethod
):
    name = "rolling_origin_forecast_backtest"
    category = "timeseries"
    version = "1.5"

    description = (
        "Evaluate naive, drift, seasonal-naive, "
        "autoregressive, simple exponential smoothing, "
        "or Holt linear-trend forecasting using "
        "expanding-window rolling-origin backtesting "
        "without look-ahead leakage."
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

        horizon = _validate_positive_integer(
            kwargs.get(
                "horizon",
                1,
            ),
            "horizon",
        )

        stride = _validate_positive_integer(
            kwargs.get(
                "stride",
                1,
            ),
            "stride",
        )

        p: int | None = None
        season_length: int | None = None
        alpha: float | None = None
        beta: float | None = None
        gamma: float | None = None

        model_parameters: dict[
            str,
            Any,
        ] = {}

        if model == "autoregressive":
            p = _validate_order(
                kwargs.get(
                    "p",
                    1,
                )
            )

            model_parameters[
                "p"
            ] = p

        if model == "seasonal_naive":
            season_length = (
                _validate_season_length(
                    kwargs.get(
                        "season_length",
                        7,
                    )
                )
            )

            model_parameters[
                "season_length"
            ] = season_length

        if (
            model
            == "simple_exponential_smoothing"
        ):
            alpha = _validate_alpha(
                kwargs.get(
                    "alpha",
                    0.2,
                )
            )

            model_parameters[
                "alpha"
            ] = alpha

        if model == "holt_linear_trend":
            alpha = _validate_alpha(
                kwargs.get(
                    "alpha",
                    0.2,
                )
            )

            beta = _validate_beta(
                kwargs.get(
                    "beta",
                    0.1,
                )
            )

            model_parameters[
                "alpha"
            ] = alpha

            model_parameters[
                "beta"
            ] = beta

        if model == "holt_winters_additive":
            alpha = _validate_alpha(
                kwargs.get(
                    "alpha",
                    0.2,
                )
            )

            beta = _validate_beta(
                kwargs.get(
                    "beta",
                    0.1,
                )
            )

            gamma = _validate_gamma(
                kwargs.get(
                    "gamma",
                    0.1,
                )
            )

            season_length = (
                _validate_season_length(
                    kwargs.get(
                        "season_length",
                        7,
                    )
                )
            )

            model_parameters.update(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "season_length": (
                        season_length
                    ),
                }
            )

        initial_train_size = kwargs.get(
            "initial_train_size"
        )

        if initial_train_size is None:
            if model == "autoregressive":
                minimum_training = (
                    2 * p + 1
                )

            elif model == "seasonal_naive":
                minimum_training = (
                    season_length
                )

            elif model == "holt_winters_additive":
                minimum_training = (
                    2 * season_length
                )

            else:
                minimum_training = 2

            initial_train_size = max(
                minimum_training,
                len(values) // 2,
            )

        initial_train_size = (
            _validate_positive_integer(
                initial_train_size,
                "initial_train_size",
            )
        )

        if initial_train_size < 2:
            raise InvalidAnalysisInputError(
                "'initial_train_size' must be at least 2."
            )

        if model == "autoregressive":
            minimum_ar_training = (
                2 * p + 1
            )

            if (
                initial_train_size
                < minimum_ar_training
            ):
                raise InvalidAnalysisInputError(
                    "AR(p) rolling-origin backtesting requires "
                    "'initial_train_size' to be at least "
                    f"{minimum_ar_training} for AR({p})."
                )

        if model == "seasonal_naive":
            if (
                initial_train_size
                < season_length
            ):
                raise InvalidAnalysisInputError(
                    "Seasonal naive rolling-origin backtesting "
                    "requires 'initial_train_size' to be at least "
                    "the seasonal length."
                )

        if model == "holt_winters_additive":
            minimum_hw_training = (
                2 * season_length
            )

            if (
                initial_train_size
                < minimum_hw_training
            ):
                raise InvalidAnalysisInputError(
                    "Additive Holt-Winters rolling-origin "
                    "backtesting requires 'initial_train_size' "
                    "to contain at least two complete seasons."
                )

        if (
            initial_train_size
            + horizon
            > len(values)
        ):
            raise InvalidAnalysisInputError(
                "Not enough observations for the requested "
                "initial training size and forecast horizon."
            )

        records: list[
            dict[str, Any]
        ] = []

        all_actual: list[float] = []
        all_forecast: list[float] = []

        origin_number = 0
        train_end = (
            initial_train_size
        )

        while (
            train_end
            + horizon
            <= len(values)
        ):
            history = values[
                :train_end
            ]

            actual = values[
                train_end:
                train_end + horizon
            ]

            forecast = (
                _forecast_from_history(
                    history,
                    model=model,
                    horizon=horizon,
                    p=p,
                    season_length=season_length,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
            )

            origin_number += 1

            for step in range(
                horizon
            ):
                target_index = (
                    train_end
                    + step
                )

                actual_value = float(
                    actual[step]
                )

                forecast_value = float(
                    forecast[step]
                )

                error = (
                    actual_value
                    - forecast_value
                )

                percentage_error = (
                    float(
                        abs(error)
                        / abs(
                            actual_value
                        )
                        * 100.0
                    )
                    if actual_value != 0
                    else None
                )

                records.append(
                    {
                        "origin": origin_number,
                        "training_start_index": 0,
                        "training_end_index": (
                            train_end - 1
                        ),
                        "training_size": (
                            train_end
                        ),
                        "origin_timestamp": (
                            time[
                                train_end - 1
                            ].isoformat()
                        ),
                        "forecast_step": (
                            step + 1
                        ),
                        "target_index": (
                            target_index
                        ),
                        "target_timestamp": (
                            time[
                                target_index
                            ].isoformat()
                        ),
                        "actual": (
                            actual_value
                        ),
                        "forecast": (
                            forecast_value
                        ),
                        "error": float(
                            error
                        ),
                        "absolute_error": float(
                            abs(error)
                        ),
                        "squared_error": float(
                            error ** 2
                        ),
                        "absolute_percentage_error": (
                            percentage_error
                        ),
                    }
                )

                all_actual.append(
                    actual_value
                )

                all_forecast.append(
                    forecast_value
                )

            train_end += stride

        if not records:
            raise InvalidAnalysisInputError(
                "The requested backtest produced no "
                "forecast evaluation observations."
            )

        actual_array = np.asarray(
            all_actual,
            dtype=float,
        )

        forecast_array = np.asarray(
            all_forecast,
            dtype=float,
        )

        metrics = _forecast_metrics(
            actual_array,
            forecast_array,
        )

        return {
            "evaluation_type": (
                "rolling_origin"
            ),
            "training_window": (
                "expanding"
            ),
            "look_ahead_leakage": False,
            "model": model,
            "model_parameters": (
                model_parameters
            ),
            "sample_size": len(
                values
            ),
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "origin_count": (
                origin_number
            ),
            "forecast_observation_count": (
                len(records)
            ),
            "metrics": metrics,
            "records": records,
        }
