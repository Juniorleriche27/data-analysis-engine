from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.forecast_ensemble import (
    _inverse_error_weights,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _validate_horizon,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    _validate_metric,
    _validate_positive_integer,
)
from data_analysis_engine.engines.timeseries.model_selection import (
    _minimum_training_size,
    _validate_models,
    _validate_parameters_by_model,
)


def _score_records(
    records: list[dict[str, Any]],
    *,
    metric: str,
) -> float | None:
    if not records:
        return None

    errors = np.asarray(
        [
            float(
                record["error"]
            )
            for record in records
        ],
        dtype=float,
    )

    if metric == "mae":
        return float(
            np.mean(
                np.abs(
                    errors
                )
            )
        )

    if metric == "rmse":
        return float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        )

    if metric == "mape":
        percentages = []

        for record in records:
            actual = float(
                record["actual"]
            )

            if actual == 0.0:
                continue

            percentages.append(
                abs(
                    float(
                        record["error"]
                    )
                    / actual
                )
                * 100.0
            )

        if not percentages:
            return None

        return float(
            np.mean(
                percentages
            )
        )

    if metric == "smape":
        percentages = []

        for record in records:
            actual = float(
                record["actual"]
            )

            forecast = float(
                record["forecast"]
            )

            denominator = (
                abs(actual)
                + abs(forecast)
            )

            if denominator == 0.0:
                continue

            percentages.append(
                200.0
                * abs(
                    actual
                    - forecast
                )
                / denominator
            )

        if not percentages:
            return None

        return float(
            np.mean(
                percentages
            )
        )

    raise InvalidAnalysisInputError(
        f"Unsupported metric: '{metric}'."
    )


def _ensemble_metrics(
    records: list[dict[str, Any]],
) -> dict[str, float | None]:
    if not records:
        raise InvalidAnalysisInputError(
            "Cannot calculate ensemble metrics "
            "without evaluation records."
        )

    errors = np.asarray(
        [
            float(
                record["error"]
            )
            for record in records
        ],
        dtype=float,
    )

    mae = float(
        np.mean(
            np.abs(
                errors
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mape_values = []

    smape_values = []

    for record in records:
        actual = float(
            record["actual"]
        )

        forecast = float(
            record["forecast"]
        )

        if actual != 0.0:
            mape_values.append(
                abs(
                    (
                        actual
                        - forecast
                    )
                    / actual
                )
                * 100.0
            )

        denominator = (
            abs(actual)
            + abs(forecast)
        )

        if denominator != 0.0:
            smape_values.append(
                200.0
                * abs(
                    actual
                    - forecast
                )
                / denominator
            )

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": (
            float(
                np.mean(
                    mape_values
                )
            )
            if mape_values
            else None
        ),
        "smape": (
            float(
                np.mean(
                    smape_values
                )
            )
            if smape_values
            else None
        ),
        "mean_error": float(
            np.mean(
                errors
            )
        ),
    }


class AdaptiveForecastEnsembleBacktest(
    AnalysisMethod
):
    name = (
        "adaptive_forecast_ensemble_backtest"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Backtest an adaptive inverse-error weighted "
        "forecast ensemble where model weights at each "
        "origin use only forecast errors already observable "
        "before that origin."
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

        models = _validate_models(
            kwargs.get(
                "models"
            )
        )

        parameters_by_model = (
            _validate_parameters_by_model(
                kwargs.get(
                    "parameters_by_model"
                ),
                models=models,
            )
        )

        metric = _validate_metric(
            kwargs.get(
                "metric",
                "mae",
            )
        )

        horizon = _validate_horizon(
            kwargs.get(
                "horizon",
                1,
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

        required_training_size = max(
            _minimum_training_size(
                model,
                parameters_by_model[
                    model
                ],
            )
            for model
            in models
        )

        initial_train_size = kwargs.get(
            "initial_train_size"
        )

        if initial_train_size is None:
            initial_train_size = max(
                required_training_size,
                len(values) // 2,
            )

        initial_train_size = (
            _validate_positive_integer(
                initial_train_size,
                "initial_train_size",
            )
        )

        if (
            initial_train_size
            < required_training_size
        ):
            raise InvalidAnalysisInputError(
                "'initial_train_size' is too small "
                "for at least one selected model. "
                f"Minimum required: "
                f"{required_training_size}."
            )

        if (
            initial_train_size
            + horizon
            > len(values)
        ):
            raise InvalidAnalysisInputError(
                "Not enough observations for the "
                "requested training size and horizon."
            )

        backtester = (
            RollingOriginForecastBacktest()
        )

        model_backtests: dict[
            str,
            dict[str, Any],
        ] = {}

        records_by_model: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        lookup_by_model: dict[
            str,
            dict[
                tuple[int, int, int],
                dict[str, Any],
            ],
        ] = {}

        for model in models:
            parameters = dict(
                parameters_by_model[
                    model
                ]
            )

            backtest = (
                backtester.execute(
                    time=list(
                        time
                    ),
                    values=(
                        values.tolist()
                    ),
                    model=model,
                    initial_train_size=(
                        initial_train_size
                    ),
                    horizon=horizon,
                    stride=stride,
                    **parameters,
                )
            )

            model_backtests[
                model
            ] = backtest

            records = list(
                backtest[
                    "records"
                ]
            )

            records_by_model[
                model
            ] = records

            lookup_by_model[
                model
            ] = {
                (
                    int(
                        record[
                            "origin"
                        ]
                    ),
                    int(
                        record[
                            "forecast_step"
                        ]
                    ),
                    int(
                        record[
                            "target_index"
                        ]
                    ),
                ): record
                for record
                in records
            }

        reference_model = models[
            0
        ]

        reference_records = (
            records_by_model[
                reference_model
            ]
        )

        ensemble_records: list[
            dict[str, Any]
        ] = []

        for reference in reference_records:
            origin = int(
                reference[
                    "origin"
                ]
            )

            step = int(
                reference[
                    "forecast_step"
                ]
            )

            target_index = int(
                reference[
                    "target_index"
                ]
            )

            training_size = int(
                reference[
                    "training_size"
                ]
            )

            key = (
                origin,
                step,
                target_index,
            )

            current_records = {}

            missing_current = False

            for model in models:
                current = (
                    lookup_by_model[
                        model
                    ].get(
                        key
                    )
                )

                if current is None:
                    missing_current = True
                    break

                current_records[
                    model
                ] = current

            if missing_current:
                continue

            historical_records = {}

            scores = []

            enough_history = True

            calibration_max_target = (
                None
            )

            for model in models:
                history = [
                    record
                    for record
                    in records_by_model[
                        model
                    ]
                    if (
                        int(
                            record[
                                "forecast_step"
                            ]
                        )
                        == step
                        and int(
                            record[
                                "target_index"
                            ]
                        )
                        < training_size
                    )
                ]

                if (
                    len(history)
                    < minimum_weight_observations
                ):
                    enough_history = False
                    break

                score = _score_records(
                    history,
                    metric=metric,
                )

                if score is None:
                    enough_history = False
                    break

                historical_records[
                    model
                ] = history

                scores.append(
                    float(
                        score
                    )
                )

                history_max_target = max(
                    int(
                        record[
                            "target_index"
                        ]
                    )
                    for record
                    in history
                )

                if (
                    calibration_max_target
                    is None
                    or history_max_target
                    > calibration_max_target
                ):
                    calibration_max_target = (
                        history_max_target
                    )

            if not enough_history:
                continue

            weights = (
                _inverse_error_weights(
                    np.asarray(
                        scores,
                        dtype=float,
                    )
                )
            )

            component_forecasts = {}

            model_weights = {}

            model_scores = {}

            ensemble_forecast = 0.0

            for (
                model,
                weight,
                score,
            ) in zip(
                models,
                weights,
                scores,
            ):
                forecast = float(
                    current_records[
                        model
                    ][
                        "forecast"
                    ]
                )

                component_forecasts[
                    model
                ] = forecast

                model_weights[
                    model
                ] = float(
                    weight
                )

                model_scores[
                    model
                ] = float(
                    score
                )

                ensemble_forecast += (
                    float(
                        weight
                    )
                    * forecast
                )

            actual = float(
                reference[
                    "actual"
                ]
            )

            error = float(
                actual
                - ensemble_forecast
            )

            ensemble_records.append(
                {
                    "origin": origin,
                    "training_size": (
                        training_size
                    ),
                    "forecast_step": (
                        step
                    ),
                    "target_index": (
                        target_index
                    ),
                    "target_timestamp": (
                        reference[
                            "target_timestamp"
                        ]
                    ),
                    "actual": actual,
                    "forecast": float(
                        ensemble_forecast
                    ),
                    "error": error,
                    "component_forecasts": (
                        component_forecasts
                    ),
                    "model_weights": (
                        model_weights
                    ),
                    "historical_model_scores": (
                        model_scores
                    ),
                    "weights_sum": float(
                        np.sum(
                            weights
                        )
                    ),
                    "weight_calibration_observations": (
                        {
                            model: len(
                                historical_records[
                                    model
                                ]
                            )
                            for model
                            in models
                        }
                    ),
                    "calibration_max_target_index": (
                        calibration_max_target
                    ),
                }
            )

        if not ensemble_records:
            raise InvalidAnalysisInputError(
                "No adaptive ensemble evaluation "
                "observations were available after the "
                "leakage-safe weighting warm-up. "
                "Increase the historical sample or lower "
                "'minimum_weight_observations'."
            )

        metrics = (
            _ensemble_metrics(
                ensemble_records
            )
        )

        per_horizon = []

        for step in range(
            1,
            horizon + 1,
        ):
            step_records = [
                record
                for record
                in ensemble_records
                if record[
                    "forecast_step"
                ]
                == step
            ]

            if not step_records:
                per_horizon.append(
                    {
                        "forecast_step": (
                            step
                        ),
                        "evaluation_observations": 0,
                        "metrics": None,
                    }
                )

                continue

            per_horizon.append(
                {
                    "forecast_step": (
                        step
                    ),
                    "evaluation_observations": (
                        len(
                            step_records
                        )
                    ),
                    "metrics": (
                        _ensemble_metrics(
                            step_records
                        )
                    ),
                }
            )

        component_full_metrics = {
            model: (
                model_backtests[
                    model
                ][
                    "metrics"
                ]
            )
            for model
            in models
        }

        return {
            "evaluation_type": (
                "adaptive_inverse_error_weighted_"
                "ensemble_backtest"
            ),
            "weighting_method": (
                "inverse_historical_out_of_sample_error"
            ),
            "sample_size": len(
                values
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "models": models,
            "models_evaluated": (
                len(
                    models
                )
            ),
            "parameters_by_model": (
                {
                    model: (
                        model_backtests[
                            model
                        ][
                            "model_parameters"
                        ]
                    )
                    for model
                    in models
                }
            ),
            "weight_metric": (
                metric
            ),
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
            "evaluation_observations": (
                len(
                    ensemble_records
                )
            ),
            "metrics": metrics,
            "per_horizon": (
                per_horizon
            ),
            "component_full_backtest_metrics": (
                component_full_metrics
            ),
            "comparable_origins": True,
            "adaptive_weights": True,
            "look_ahead_leakage": False,
            "weight_calibration_rule": (
                "only_model_errors_with_target_index_"
                "before_current_training_origin"
            ),
            "records": (
                ensemble_records
            ),
        }
