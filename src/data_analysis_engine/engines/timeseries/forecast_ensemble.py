from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
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
from data_analysis_engine.engines.timeseries.model_selection import (
    ForecastModelSelection,
)


def _inverse_error_weights(
    scores: np.ndarray,
    *,
    zero_tolerance: float = 1e-12,
) -> np.ndarray:
    scores = np.asarray(
        scores,
        dtype=float,
    )

    if (
        scores.ndim != 1
        or len(scores) == 0
    ):
        raise InvalidAnalysisInputError(
            "Ensemble scores must be a non-empty "
            "one-dimensional sequence."
        )

    if (
        not np.all(
            np.isfinite(
                scores
            )
        )
        or np.any(
            scores < 0.0
        )
    ):
        raise InvalidAnalysisInputError(
            "Ensemble scores must be finite "
            "and non-negative."
        )

    near_zero = (
        scores
        <= zero_tolerance
    )

    if np.any(
        near_zero
    ):
        weights = np.zeros(
            len(scores),
            dtype=float,
        )

        count = int(
            np.sum(
                near_zero
            )
        )

        weights[
            near_zero
        ] = (
            1.0
            / count
        )

        return weights

    inverse = (
        1.0
        / scores
    )

    return (
        inverse
        / np.sum(
            inverse
        )
    )


class WeightedForecastEnsemble(
    AnalysisMethod
):
    name = "weighted_forecast_ensemble"
    category = "timeseries"
    version = "1.0"

    description = (
        "Combine multiple forecasting models using "
        "inverse out-of-sample error weights obtained "
        "from comparable rolling-origin evaluation."
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

        selector_kwargs = {
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
        }

        if (
            kwargs.get(
                "initial_train_size"
            )
            is not None
        ):
            selector_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        selection = (
            ForecastModelSelection()
            .execute(
                **selector_kwargs
            )
        )

        candidates = list(
            selection[
                "candidates"
            ]
        )

        top_k_value = kwargs.get(
            "top_k"
        )

        if top_k_value is None:
            top_k = len(
                candidates
            )
        else:
            top_k = (
                _validate_positive_integer(
                    top_k_value,
                    "top_k",
                )
            )

        if (
            top_k
            > len(
                candidates
            )
        ):
            raise InvalidAnalysisInputError(
                "'top_k' cannot exceed the number "
                "of evaluated models."
            )

        selected_candidates = (
            candidates[
                :top_k
            ]
        )

        scores = np.asarray(
            [
                float(
                    candidate[
                        "score"
                    ]
                )
                for candidate
                in selected_candidates
            ],
            dtype=float,
        )

        weights = (
            _inverse_error_weights(
                scores
            )
        )

        component_forecasts = []

        forecast_matrix = []

        for candidate, weight in zip(
            selected_candidates,
            weights,
        ):
            model = candidate[
                "model"
            ]

            parameters = dict(
                candidate[
                    "model_parameters"
                ]
            )

            forecast_values = (
                _forecast_from_history(
                    values,
                    model=model,
                    horizon=horizon,
                    p=parameters.get(
                        "p"
                    ),
                    season_length=(
                        parameters.get(
                            "season_length"
                        )
                    ),
                    alpha=parameters.get(
                        "alpha"
                    ),
                    beta=parameters.get(
                        "beta"
                    ),
                    gamma=parameters.get(
                        "gamma"
                    ),
                )
            )

            forecast_matrix.append(
                forecast_values
            )

            component_forecasts.append(
                {
                    "model": model,
                    "rank": candidate[
                        "rank"
                    ],
                    "weight": float(
                        weight
                    ),
                    "selection_score": float(
                        candidate[
                            "score"
                        ]
                    ),
                    "model_parameters": (
                        parameters
                    ),
                    "metrics": (
                        candidate[
                            "metrics"
                        ]
                    ),
                    "forecast_values": [
                        float(
                            value
                        )
                        for value
                        in forecast_values
                    ],
                }
            )

        matrix = np.asarray(
            forecast_matrix,
            dtype=float,
        )

        ensemble_forecast = (
            np.sum(
                matrix
                * weights[
                    :,
                    None
                ],
                axis=0,
            )
        )

        forecast_timestamps = (
            _future_timestamps(
                time,
                interval_ns,
                horizon,
            )
        )

        forecasts = []

        for step in range(
            1,
            horizon + 1,
        ):
            component_values = {
                component[
                    "model"
                ]: component[
                    "forecast_values"
                ][
                    step - 1
                ]
                for component
                in component_forecasts
            }

            forecasts.append(
                {
                    "forecast_step": (
                        step
                    ),
                    "timestamp": (
                        forecast_timestamps[
                            step - 1
                        ]
                    ),
                    "forecast": float(
                        ensemble_forecast[
                            step - 1
                        ]
                    ),
                    "component_forecasts": (
                        component_values
                    ),
                }
            )

        return {
            "forecast_type": (
                "inverse_error_weighted_ensemble"
            ),
            "weighting_method": (
                "inverse_out_of_sample_error"
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
            "models_in_ensemble": (
                len(
                    selected_candidates
                )
            ),
            "top_k": top_k,
            "best_individual_model": (
                selection[
                    "best_model"
                ]
            ),
            "best_individual_score": (
                selection[
                    "best_score"
                ]
            ),
            "component_forecasts": (
                component_forecasts
            ),
            "model_ranking": (
                selection[
                    "candidates"
                ]
            ),
            "weights_sum": float(
                np.sum(
                    weights
                )
            ),
            "comparable_origins": True,
            "look_ahead_leakage": False,
            "refit_components_on_full_history": True,
            "forecast_timestamps": (
                forecast_timestamps
            ),
            "forecast_values": [
                float(
                    value
                )
                for value
                in ensemble_forecast
            ],
            "forecasts": forecasts,
        }
