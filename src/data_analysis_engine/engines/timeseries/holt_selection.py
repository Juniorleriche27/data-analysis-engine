from typing import Any

import numpy as np

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
)
from data_analysis_engine.engines.timeseries.holt import (
    _validate_beta,
)


SUPPORTED_SELECTION_METRICS = (
    "mae",
    "rmse",
    "mape",
    "smape",
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


def _validate_metric(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'metric' must be a string."
        )

    metric = value.lower()

    if metric not in SUPPORTED_SELECTION_METRICS:
        raise InvalidAnalysisInputError(
            "'metric' must be one of: "
            + ", ".join(
                SUPPORTED_SELECTION_METRICS
            )
            + "."
        )

    return metric


def _validate_parameter_candidates(
    value: Any,
    *,
    name: str,
    validator,
    default: list[float],
) -> list[float]:
    if value is None:
        value = default

    if (
        not isinstance(
            value,
            (
                list,
                tuple,
                np.ndarray,
            ),
        )
        or isinstance(
            value,
            (str, bytes),
        )
    ):
        raise InvalidAnalysisInputError(
            f"'{name}' must be a sequence."
        )

    if len(value) == 0:
        raise InvalidAnalysisInputError(
            f"'{name}' must contain at least one value."
        )

    candidates = [
        validator(candidate)
        for candidate in value
    ]

    if len(
        set(candidates)
    ) != len(candidates):
        raise InvalidAnalysisInputError(
            f"'{name}' must not contain duplicate values."
        )

    return candidates


class HoltParameterSelection(
    AnalysisMethod
):
    name = "holt_parameter_selection"
    category = "timeseries"
    version = "1.0"

    description = (
        "Select Holt linear-trend alpha and beta parameters "
        "using comparable rolling-origin out-of-sample "
        "forecast evaluation."
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

        alphas = _validate_parameter_candidates(
            kwargs.get(
                "alphas"
            ),
            name="alphas",
            validator=_validate_alpha,
            default=[
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
        )

        betas = _validate_parameter_candidates(
            kwargs.get(
                "betas"
            ),
            name="betas",
            validator=_validate_beta,
            default=[
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
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
                backtest = backtester.execute(
                    time=list(time),
                    values=values.tolist(),
                    model="holt_linear_trend",
                    alpha=alpha,
                    beta=beta,
                    initial_train_size=(
                        initial_train_size
                    ),
                    horizon=horizon,
                    stride=stride,
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
                        "score": (
                            float(score)
                            if score is not None
                            else None
                        ),
                        "score_defined": (
                            score is not None
                        ),
                        "metrics": metrics,
                        "origin_count": backtest[
                            "origin_count"
                        ],
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
                "was undefined for every Holt candidate."
            )

        ranking = sorted(
            rankable,
            key=lambda candidate: (
                candidate["score"],
                candidate["alpha"],
                candidate["beta"],
            ),
        )

        rank_by_parameters = {
            (
                candidate["alpha"],
                candidate["beta"],
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
            ] = rank_by_parameters.get(
                (
                    candidate["alpha"],
                    candidate["beta"],
                )
            )

            ranked_candidates.append(
                ranked
            )

        ranked_candidates.sort(
            key=lambda candidate: (
                candidate["rank"]
                if candidate["rank"] is not None
                else float("inf"),
                candidate["alpha"],
                candidate["beta"],
            )
        )

        best = ranking[0]

        return {
            "selection_type": (
                "rolling_origin_holt_parameter_selection"
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
            "alpha_candidates": len(
                alphas
            ),
            "beta_candidates": len(
                betas
            ),
            "parameter_combinations_evaluated": (
                len(alphas)
                * len(betas)
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
            "best_parameters": {
                "alpha": float(
                    best["alpha"]
                ),
                "beta": float(
                    best["beta"]
                ),
            },
            "best_score": float(
                best["score"]
            ),
            "best_metrics": best[
                "metrics"
            ],
            "candidates": (
                ranked_candidates
            ),
        }
