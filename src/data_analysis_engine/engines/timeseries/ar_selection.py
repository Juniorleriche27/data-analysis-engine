from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
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


class AutoregressiveOrderSelection(
    AnalysisMethod
):
    name = "autoregressive_order_selection"
    category = "timeseries"
    version = "1.0"

    description = (
        "Select the AR(p) order using comparable "
        "rolling-origin out-of-sample backtests."
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

        min_p = _validate_positive_integer(
            kwargs.get(
                "min_p",
                1,
            ),
            "min_p",
        )

        max_p = _validate_positive_integer(
            kwargs.get(
                "max_p",
                3,
            ),
            "max_p",
        )

        if max_p < min_p:
            raise InvalidAnalysisInputError(
                "'max_p' must be greater than "
                "or equal to 'min_p'."
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

        metric = _validate_metric(
            kwargs.get(
                "metric",
                "mae",
            )
        )

        minimum_training_size = (
            2 * max_p + 1
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
                "Comparable AR order selection requires "
                "'initial_train_size' to be at least "
                f"{minimum_training_size} when "
                f"'max_p' is {max_p}."
            )

        if (
            initial_train_size
            + horizon
            > len(values)
        ):
            raise InvalidAnalysisInputError(
                "Not enough observations for the requested "
                "AR search range, training size, and horizon."
            )

        backtester = (
            RollingOriginForecastBacktest()
        )

        candidates: list[
            dict[str, Any]
        ] = []

        for p in range(
            min_p,
            max_p + 1,
        ):
            backtest = backtester.execute(
                time=list(time),
                values=values.tolist(),
                model="autoregressive",
                p=p,
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

            candidate = {
                "p": p,
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

            candidates.append(
                candidate
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
                "was undefined for every AR candidate."
            )

        ranking = sorted(
            rankable,
            key=lambda candidate: (
                candidate["score"],
                candidate["p"],
            ),
        )

        ranked_candidates: list[
            dict[str, Any]
        ] = []

        rank_by_p = {
            candidate["p"]: rank
            for rank, candidate in enumerate(
                ranking,
                start=1,
            )
        }

        for candidate in candidates:
            ranked = dict(
                candidate
            )

            ranked[
                "rank"
            ] = rank_by_p.get(
                candidate["p"]
            )

            ranked_candidates.append(
                ranked
            )

        ranked_candidates.sort(
            key=lambda candidate: (
                candidate["rank"]
                if candidate["rank"] is not None
                else float("inf"),
                candidate["p"],
            )
        )

        best = ranking[0]

        return {
            "selection_type": (
                "rolling_origin_ar_order_selection"
            ),
            "selection_objective": (
                "minimize_out_of_sample_metric"
            ),
            "look_ahead_leakage": False,
            "comparable_origins": True,
            "sample_size": len(values),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "min_p": min_p,
            "max_p": max_p,
            "orders_evaluated": (
                max_p - min_p + 1
            ),
            "metric": metric,
            "initial_train_size": (
                initial_train_size
            ),
            "horizon": horizon,
            "stride": stride,
            "best_order": best[
                "p"
            ],
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
