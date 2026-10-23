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


def _validate_alpha_candidates(
    value: Any,
) -> list[float]:
    if value is None:
        value = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

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
            "'alphas' must be a sequence."
        )

    if len(value) == 0:
        raise InvalidAnalysisInputError(
            "'alphas' must contain at least one value."
        )

    alphas = [
        _validate_alpha(
            alpha
        )
        for alpha in value
    ]

    if len(
        set(
            alphas
        )
    ) != len(alphas):
        raise InvalidAnalysisInputError(
            "'alphas' must not contain duplicate values."
        )

    return alphas


class SimpleExponentialSmoothingAlphaSelection(
    AnalysisMethod
):
    name = (
        "simple_exponential_smoothing_alpha_selection"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Select the simple exponential smoothing alpha "
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

        alphas = (
            _validate_alpha_candidates(
                kwargs.get(
                    "alphas"
                )
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
            backtest = backtester.execute(
                time=list(time),
                values=values.tolist(),
                model=(
                    "simple_exponential_smoothing"
                ),
                alpha=alpha,
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
                "was undefined for every alpha candidate."
            )

        ranking = sorted(
            rankable,
            key=lambda candidate: (
                candidate["score"],
                candidate["alpha"],
            ),
        )

        rank_by_alpha = {
            candidate[
                "alpha"
            ]: rank
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
            ] = rank_by_alpha.get(
                candidate[
                    "alpha"
                ]
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
            )
        )

        best = ranking[0]

        return {
            "selection_type": (
                "rolling_origin_ses_alpha_selection"
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
            "alphas_evaluated": len(
                alphas
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
