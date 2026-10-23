from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.autoregression import (
    _validate_order,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
    SUPPORTED_MODELS,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _validate_season_length,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    _validate_metric,
    _validate_positive_integer,
)


DEFAULT_MODELS = (
    "naive",
    "drift",
    "autoregressive",
    "simple_exponential_smoothing",
    "holt_linear_trend",
)


MODEL_PARAMETER_NAMES = {
    "naive": set(),
    "drift": set(),
    "seasonal_naive": {
        "season_length",
    },
    "autoregressive": {
        "p",
    },
    "simple_exponential_smoothing": {
        "alpha",
    },
    "holt_linear_trend": {
        "alpha",
        "beta",
    },
    "holt_winters_additive": {
        "alpha",
        "beta",
        "gamma",
        "season_length",
    },
}


def _validate_models(
    value: Any,
) -> list[str]:
    if value is None:
        models = list(
            DEFAULT_MODELS
        )
    else:
        if (
            not isinstance(
                value,
                (
                    list,
                    tuple,
                ),
            )
            or isinstance(
                value,
                str,
            )
            or len(value) == 0
        ):
            raise InvalidAnalysisInputError(
                "'models' must be a non-empty "
                "list or tuple."
            )

        models = list(
            value
        )

    for model in models:
        if (
            not isinstance(
                model,
                str,
            )
            or model
            not in SUPPORTED_MODELS
        ):
            raise InvalidAnalysisInputError(
                "Every model must be one of: "
                + ", ".join(
                    SUPPORTED_MODELS
                )
                + "."
            )

    if (
        len(
            set(
                models
            )
        )
        != len(
            models
        )
    ):
        raise InvalidAnalysisInputError(
            "'models' must not contain duplicates."
        )

    return models


def _validate_parameters_by_model(
    value: Any,
    *,
    models: list[str],
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {
            model: {}
            for model in models
        }

    if not isinstance(
        value,
        dict,
    ):
        raise InvalidAnalysisInputError(
            "'parameters_by_model' must be a dictionary."
        )

    unknown_models = (
        set(
            value
        )
        - set(
            models
        )
    )

    if unknown_models:
        raise InvalidAnalysisInputError(
            "'parameters_by_model' contains models "
            "that are not included in 'models': "
            + ", ".join(
                sorted(
                    unknown_models
                )
            )
            + "."
        )

    result = {}

    for model in models:
        parameters = value.get(
            model,
            {},
        )

        if not isinstance(
            parameters,
            dict,
        ):
            raise InvalidAnalysisInputError(
                f"Parameters for model '{model}' "
                "must be a dictionary."
            )

        unsupported = (
            set(
                parameters
            )
            - MODEL_PARAMETER_NAMES[
                model
            ]
        )

        if unsupported:
            raise InvalidAnalysisInputError(
                f"Unsupported parameter(s) for "
                f"model '{model}': "
                + ", ".join(
                    sorted(
                        unsupported
                    )
                )
                + "."
            )

        result[
            model
        ] = dict(
            parameters
        )

    return result


def _minimum_training_size(
    model: str,
    parameters: dict[str, Any],
) -> int:
    if model == "naive":
        return 1

    if model == "drift":
        return 2

    if model == "seasonal_naive":
        season_length = (
            _validate_season_length(
                parameters.get(
                    "season_length",
                    7,
                )
            )
        )

        return season_length

    if model == "autoregressive":
        order = _validate_order(
            parameters.get(
                "p",
                1,
            )
        )

        return (
            2 * order
            + 1
        )

    if (
        model
        == "simple_exponential_smoothing"
    ):
        return 2

    if model == "holt_linear_trend":
        return 2

    if model == "holt_winters_additive":
        season_length = (
            _validate_season_length(
                parameters.get(
                    "season_length",
                    7,
                )
            )
        )

        return (
            2
            * season_length
        )

    raise InvalidAnalysisInputError(
        f"Unsupported forecast model: '{model}'."
    )


class ForecastModelSelection(
    AnalysisMethod
):
    name = "forecast_model_selection"
    category = "timeseries"
    version = "1.0"

    description = (
        "Compare multiple forecasting models using "
        "the same rolling-origin out-of-sample "
        "evaluation origins and select the model "
        "with the lowest forecast error."
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

        required_training_size = max(
            _minimum_training_size(
                model,
                parameters_by_model[
                    model
                ],
            )
            for model in models
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

        candidates: list[
            dict[str, Any]
        ] = []

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

            score = backtest[
                "metrics"
            ][
                metric
            ]

            candidates.append(
                {
                    "model": model,
                    "model_parameters": (
                        backtest[
                            "model_parameters"
                        ]
                    ),
                    "score": (
                        float(
                            score
                        )
                        if score
                        is not None
                        else None
                    ),
                    "score_defined": (
                        score
                        is not None
                    ),
                    "metrics": (
                        backtest[
                            "metrics"
                        ]
                    ),
                    "origin_count": (
                        backtest[
                            "origin_count"
                        ]
                    ),
                    "forecast_observation_count": (
                        backtest[
                            "forecast_observation_count"
                        ]
                    ),
                }
            )

        rankable = [
            candidate
            for candidate in candidates
            if candidate[
                "score_defined"
            ]
        ]

        if not rankable:
            raise InvalidAnalysisInputError(
                f"The selection metric '{metric}' "
                "was undefined for every model."
            )

        ranking = sorted(
            rankable,
            key=lambda candidate: (
                candidate[
                    "score"
                ],
                candidate[
                    "model"
                ],
            ),
        )

        rank_by_model = {
            candidate[
                "model"
            ]: rank
            for rank, candidate
            in enumerate(
                ranking,
                start=1,
            )
        }

        ranked_candidates = []

        for candidate in candidates:
            ranked = dict(
                candidate
            )

            ranked[
                "rank"
            ] = rank_by_model[
                candidate[
                    "model"
                ]
            ]

            ranked_candidates.append(
                ranked
            )

        ranked_candidates.sort(
            key=lambda candidate: (
                candidate[
                    "rank"
                ],
                candidate[
                    "model"
                ],
            )
        )

        best = ranking[
            0
        ]

        return {
            "selection_type": (
                "rolling_origin_forecast_model_selection"
            ),
            "selection_objective": (
                "minimize_out_of_sample_metric"
            ),
            "sample_size": len(
                values
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "models_evaluated": len(
                models
            ),
            "models": models,
            "metric": metric,
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "best_model": (
                best[
                    "model"
                ]
            ),
            "best_model_parameters": (
                best[
                    "model_parameters"
                ]
            ),
            "best_score": float(
                best[
                    "score"
                ]
            ),
            "best_metrics": (
                best[
                    "metrics"
                ]
            ),
            "candidates": (
                ranked_candidates
            ),
        }
