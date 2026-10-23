from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.exponential_smoothing import (
    _validate_alpha,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _validate_season_length,
)
from data_analysis_engine.engines.timeseries.holt import (
    _validate_beta,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    _validate_metric,
    _validate_parameter_candidates,
    _validate_positive_integer,
)
from data_analysis_engine.engines.timeseries.holt_winters import (
    _validate_gamma,
)


class HoltWintersAdditiveParameterSelection(
    AnalysisMethod
):
    name = (
        "holt_winters_additive_parameter_selection"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Select additive Holt-Winters alpha, beta, and gamma "
        "parameters using comparable rolling-origin "
        "out-of-sample forecast evaluation."
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

        season_length = (
            _validate_season_length(
                kwargs.get(
                    "season_length",
                    7,
                )
            )
        )

        alphas = (
            _validate_parameter_candidates(
                kwargs.get(
                    "alphas"
                ),
                name="alphas",
                validator=_validate_alpha,
                default=[
                    0.2,
                    0.5,
                    0.8,
                    1.0,
                ],
            )
        )

        betas = (
            _validate_parameter_candidates(
                kwargs.get(
                    "betas"
                ),
                name="betas",
                validator=_validate_beta,
                default=[
                    0.2,
                    0.5,
                    0.8,
                    1.0,
                ],
            )
        )

        gammas = (
            _validate_parameter_candidates(
                kwargs.get(
                    "gammas"
                ),
                name="gammas",
                validator=_validate_gamma,
                default=[
                    0.2,
                    0.5,
                    0.8,
                    1.0,
                ],
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

        minimum_training_size = (
            2 * season_length
        )

        initial_train_size = kwargs.get(
            "initial_train_size"
        )

        if initial_train_size is None:
            initial_train_size = max(
                minimum_training_size,
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
            < minimum_training_size
        ):
            raise InvalidAnalysisInputError(
                "Additive Holt-Winters parameter selection "
                "requires 'initial_train_size' to contain "
                "at least two complete seasons."
            )

        if (
            initial_train_size
            + horizon
            > len(values)
        ):
            raise InvalidAnalysisInputError(
                "Not enough observations for the requested "
                "training size and forecast horizon."
            )

        backtester = (
            RollingOriginForecastBacktest()
        )

        candidates: list[
            dict[str, Any]
        ] = []

        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    backtest = (
                        backtester.execute(
                            time=list(
                                time
                            ),
                            values=(
                                values.tolist()
                            ),
                            model=(
                                "holt_winters_additive"
                            ),
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            season_length=(
                                season_length
                            ),
                            initial_train_size=(
                                initial_train_size
                            ),
                            horizon=horizon,
                            stride=stride,
                        )
                    )

                    metrics = backtest[
                        "metrics"
                    ]

                    score = metrics[
                        metric
                    ]

                    candidates.append(
                        {
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma,
                            "score": (
                                float(score)
                                if score
                                is not None
                                else None
                            ),
                            "score_defined": (
                                score
                                is not None
                            ),
                            "metrics": (
                                metrics
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
                "was undefined for every "
                "Holt-Winters candidate."
            )

        ranking = sorted(
            rankable,
            key=lambda candidate: (
                candidate[
                    "score"
                ],
                candidate[
                    "alpha"
                ],
                candidate[
                    "beta"
                ],
                candidate[
                    "gamma"
                ],
            ),
        )

        rank_by_parameters = {
            (
                candidate[
                    "alpha"
                ],
                candidate[
                    "beta"
                ],
                candidate[
                    "gamma"
                ],
            ): rank
            for rank, candidate in enumerate(
                ranking,
                start=1,
            )
        }

        ranked_candidates: list[
            dict[str, Any]
        ] = []

        for candidate in candidates:
            ranked = dict(
                candidate
            )

            ranked[
                "rank"
            ] = rank_by_parameters[
                (
                    candidate[
                        "alpha"
                    ],
                    candidate[
                        "beta"
                    ],
                    candidate[
                        "gamma"
                    ],
                )
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
                    "alpha"
                ],
                candidate[
                    "beta"
                ],
                candidate[
                    "gamma"
                ],
            )
        )

        best = ranking[
            0
        ]

        return {
            "selection_type": (
                "rolling_origin_holt_winters_"
                "additive_parameter_selection"
            ),
            "selection_objective": (
                "minimize_out_of_sample_metric"
            ),
            "look_ahead_leakage": False,
            "comparable_origins": True,
            "sample_size": len(
                values
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "season_length": (
                season_length
            ),
            "alpha_candidates": len(
                alphas
            ),
            "beta_candidates": len(
                betas
            ),
            "gamma_candidates": len(
                gammas
            ),
            "parameter_combinations_evaluated": (
                len(alphas)
                * len(betas)
                * len(gammas)
            ),
            "metric": metric,
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "best_alpha": float(
                best["alpha"]
            ),
            "best_beta": float(
                best["beta"]
            ),
            "best_gamma": float(
                best["gamma"]
            ),
            "best_parameters": {
                "alpha": float(
                    best["alpha"]
                ),
                "beta": float(
                    best["beta"]
                ),
                "gamma": float(
                    best["gamma"]
                ),
                "season_length": (
                    season_length
                ),
            },
            "best_score": float(
                best["score"]
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
