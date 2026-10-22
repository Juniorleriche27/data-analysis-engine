from math import sqrt
from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
)


SUPPORTED_BASELINES = (
    "naive",
    "drift",
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
    version = "1.0"

    description = (
        "Evaluate naive or drift forecasting using "
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

        if model not in SUPPORTED_BASELINES:
            raise InvalidAnalysisInputError(
                "'model' must be one of: "
                + ", ".join(
                    SUPPORTED_BASELINES
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

        initial_train_size = kwargs.get(
            "initial_train_size"
        )

        if initial_train_size is None:
            initial_train_size = max(
                2,
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
            "sample_size": len(values),
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "origin_count": origin_number,
            "forecast_observation_count": (
                len(records)
            ),
            "metrics": metrics,
            "records": records,
        }
