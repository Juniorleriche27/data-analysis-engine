from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.ensemble_backtest import (
    AdaptiveForecastEnsembleBacktest,
    _ensemble_metrics,
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


class ForecastStrategySelection(
    AnalysisMethod
):
    name = "forecast_strategy_selection"
    category = "timeseries"
    version = "1.0"

    description = (
        "Compare individual forecasting models and an "
        "adaptive weighted ensemble on exactly the same "
        "leakage-safe out-of-sample evaluation observations."
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

        if len(models) < 2:
            raise InvalidAnalysisInputError(
                "Forecast strategy selection requires "
                "at least two models so that an ensemble "
                "comparison is meaningful."
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

        ensemble = (
            AdaptiveForecastEnsembleBacktest()
            .execute(
                time=list(
                    time
                ),
                values=values.tolist(),
                models=models,
                parameters_by_model=(
                    parameters_by_model
                ),
                initial_train_size=(
                    initial_train_size
                ),
                horizon=horizon,
                stride=stride,
                metric=metric,
                minimum_weight_observations=(
                    minimum_weight_observations
                ),
            )
        )

        common_keys = {
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
            )
            for record
            in ensemble[
                "records"
            ]
        }

        if not common_keys:
            raise InvalidAnalysisInputError(
                "No common evaluation observations "
                "were available for strategy comparison."
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

            common_records = [
                record
                for record
                in backtest[
                    "records"
                ]
                if (
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
                )
                in common_keys
            ]

            if (
                len(
                    common_records
                )
                != len(
                    common_keys
                )
            ):
                raise InvalidAnalysisInputError(
                    f"Model '{model}' does not have "
                    "the complete common evaluation "
                    "window required for a fair "
                    "strategy comparison."
                )

            metrics = (
                _ensemble_metrics(
                    common_records
                )
            )

            score = metrics[
                metric
            ]

            candidates.append(
                {
                    "strategy": (
                        "single_model"
                    ),
                    "strategy_name": (
                        model
                    ),
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
                    "metrics": metrics,
                    "evaluation_observations": (
                        len(
                            common_records
                        )
                    ),
                }
            )

        ensemble_metrics = (
            ensemble[
                "metrics"
            ]
        )

        ensemble_score = (
            ensemble_metrics[
                metric
            ]
        )

        candidates.append(
            {
                "strategy": (
                    "adaptive_ensemble"
                ),
                "strategy_name": (
                    "adaptive_ensemble"
                ),
                "model": None,
                "model_parameters": None,
                "score": (
                    float(
                        ensemble_score
                    )
                    if ensemble_score
                    is not None
                    else None
                ),
                "score_defined": (
                    ensemble_score
                    is not None
                ),
                "metrics": (
                    ensemble_metrics
                ),
                "evaluation_observations": (
                    ensemble[
                        "evaluation_observations"
                    ]
                ),
                "ensemble_models": (
                    models
                ),
                "weighting_method": (
                    ensemble[
                        "weighting_method"
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
                f"The strategy selection metric "
                f"'{metric}' was undefined for "
                "every strategy."
            )

        ranking = sorted(
            rankable,
            key=lambda candidate: (
                candidate[
                    "score"
                ],
                (
                    0
                    if candidate[
                        "strategy"
                    ]
                    == "single_model"
                    else 1
                ),
                candidate[
                    "strategy_name"
                ],
            ),
        )

        ranked_candidates = []

        for rank, candidate in enumerate(
            ranking,
            start=1,
        ):
            ranked = dict(
                candidate
            )

            ranked[
                "rank"
            ] = rank

            ranked_candidates.append(
                ranked
            )

        best = ranked_candidates[
            0
        ]

        common_target_indices = [
            key[
                2
            ]
            for key in common_keys
        ]

        return {
            "selection_type": (
                "fair_out_of_sample_forecast_"
                "strategy_selection"
            ),
            "selection_objective": (
                "minimize_common_window_"
                "out_of_sample_metric"
            ),
            "sample_size": len(
                values
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "metric": metric,
            "models": models,
            "models_evaluated": (
                len(
                    models
                )
            ),
            "strategies_evaluated": (
                len(
                    candidates
                )
            ),
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
            "common_evaluation_observations": (
                len(
                    common_keys
                )
            ),
            "common_target_index_min": (
                min(
                    common_target_indices
                )
            ),
            "common_target_index_max": (
                max(
                    common_target_indices
                )
            ),
            "same_evaluation_window": True,
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "tie_break_rule": (
                "prefer_single_model_when_scores_equal"
            ),
            "best_strategy": (
                best[
                    "strategy"
                ]
            ),
            "best_strategy_name": (
                best[
                    "strategy_name"
                ]
            ),
            "best_model": (
                best.get(
                    "model"
                )
            ),
            "best_model_parameters": (
                best.get(
                    "model_parameters"
                )
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
            "adaptive_ensemble_backtest": {
                "metrics": (
                    ensemble[
                        "metrics"
                    ]
                ),
                "evaluation_observations": (
                    ensemble[
                        "evaluation_observations"
                    ]
                ),
            },
        }
