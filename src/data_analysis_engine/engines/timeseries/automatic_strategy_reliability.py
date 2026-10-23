from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.automatic_forecast_reliability import (
    _calibration_status,
    _validate_calibration_tolerance,
)
from data_analysis_engine.engines.timeseries.automatic_strategy_intervals import (
    AutomaticStrategyForecastWithIntervals,
)
from data_analysis_engine.engines.timeseries.conformal_intervals import (
    _finite_sample_conformal_radius,
)
from data_analysis_engine.engines.timeseries.ensemble_backtest import (
    AdaptiveForecastEnsembleBacktest,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
)
from data_analysis_engine.engines.timeseries.forecast_intervals import (
    _validate_coverage,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _validate_horizon,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    _validate_metric,
    _validate_positive_integer,
)


def _sequential_interval_calibration(
    records: list[dict[str, Any]],
    *,
    horizon: int,
    coverage: float,
    minimum_calibration_observations: int,
) -> dict[str, Any]:
    records_by_step = {
        step: [
            record
            for record in records
            if int(
                record[
                    "forecast_step"
                ]
            )
            == step
        ]
        for step in range(
            1,
            horizon + 1,
        )
    }

    validation_records: list[
        dict[str, Any]
    ] = []

    for current in records:
        step = int(
            current[
                "forecast_step"
            ]
        )

        training_size = int(
            current[
                "training_size"
            ]
        )

        historical = [
            record
            for record in records_by_step[
                step
            ]
            if int(
                record[
                    "target_index"
                ]
            )
            < training_size
        ]

        if (
            len(
                historical
            )
            < minimum_calibration_observations
        ):
            continue

        absolute_errors = np.asarray(
            [
                abs(
                    float(
                        record[
                            "error"
                        ]
                    )
                )
                for record
                in historical
            ],
            dtype=float,
        )

        conformal = (
            _finite_sample_conformal_radius(
                absolute_errors,
                coverage=coverage,
            )
        )

        radius = float(
            conformal[
                "radius"
            ]
        )

        forecast = float(
            current[
                "forecast"
            ]
        )

        actual = float(
            current[
                "actual"
            ]
        )

        lower_bound = float(
            forecast
            - radius
        )

        upper_bound = float(
            forecast
            + radius
        )

        covered = bool(
            lower_bound
            <= actual
            <= upper_bound
        )

        calibration_max_target_index = max(
            int(
                record[
                    "target_index"
                ]
            )
            for record
            in historical
        )

        validation_records.append(
            {
                "origin": int(
                    current[
                        "origin"
                    ]
                ),
                "training_size": (
                    training_size
                ),
                "forecast_step": (
                    step
                ),
                "target_index": int(
                    current[
                        "target_index"
                    ]
                ),
                "target_timestamp": (
                    current[
                        "target_timestamp"
                    ]
                ),
                "actual": actual,
                "forecast": forecast,
                "lower_bound": (
                    lower_bound
                ),
                "upper_bound": (
                    upper_bound
                ),
                "radius": radius,
                "interval_width": float(
                    2.0
                    * radius
                ),
                "covered": covered,
                "calibration_observations": (
                    conformal[
                        "sample_size"
                    ]
                ),
                "conformal_rank": (
                    conformal[
                        "rank"
                    ]
                ),
                "calibration_max_target_index": (
                    calibration_max_target_index
                ),
            }
        )

    if not validation_records:
        raise InvalidAnalysisInputError(
            "No historical interval validation "
            "observations were available after the "
            "leakage-safe calibration warm-up."
        )

    covered = np.asarray(
        [
            record[
                "covered"
            ]
            for record
            in validation_records
        ],
        dtype=float,
    )

    widths = np.asarray(
        [
            record[
                "interval_width"
            ]
            for record
            in validation_records
        ],
        dtype=float,
    )

    empirical_coverage = float(
        np.mean(
            covered
        )
    )

    coverage_gap = float(
        empirical_coverage
        - coverage
    )

    per_horizon = []

    for step in range(
        1,
        horizon + 1,
    ):
        step_records = [
            record
            for record
            in validation_records
            if record[
                "forecast_step"
            ]
            == step
        ]

        if not step_records:
            per_horizon.append(
                {
                    "forecast_step": step,
                    "validation_observations": 0,
                    "empirical_coverage": None,
                    "coverage_gap": None,
                    "absolute_coverage_error": None,
                    "average_interval_width": None,
                }
            )
            continue

        step_covered = np.asarray(
            [
                record[
                    "covered"
                ]
                for record
                in step_records
            ],
            dtype=float,
        )

        step_widths = np.asarray(
            [
                record[
                    "interval_width"
                ]
                for record
                in step_records
            ],
            dtype=float,
        )

        step_coverage = float(
            np.mean(
                step_covered
            )
        )

        step_gap = float(
            step_coverage
            - coverage
        )

        per_horizon.append(
            {
                "forecast_step": (
                    step
                ),
                "validation_observations": (
                    len(
                        step_records
                    )
                ),
                "empirical_coverage": (
                    step_coverage
                ),
                "coverage_gap": (
                    step_gap
                ),
                "absolute_coverage_error": float(
                    abs(
                        step_gap
                    )
                ),
                "average_interval_width": float(
                    np.mean(
                        step_widths
                    )
                ),
            }
        )

    return {
        "validation_observations": (
            len(
                validation_records
            )
        ),
        "covered_observations": int(
            np.sum(
                covered
            )
        ),
        "empirical_coverage": (
            empirical_coverage
        ),
        "coverage_gap": (
            coverage_gap
        ),
        "absolute_coverage_error": float(
            abs(
                coverage_gap
            )
        ),
        "average_interval_width": float(
            np.mean(
                widths
            )
        ),
        "median_interval_width": float(
            np.median(
                widths
            )
        ),
        "per_horizon": (
            per_horizon
        ),
        "records": (
            validation_records
        ),
    }


class AutomaticStrategyForecastReliability(
    AnalysisMethod
):
    name = (
        "automatic_strategy_forecast_reliability"
    )
    category = "timeseries"
    version = "1.0"

    description = (
        "Select the best forecasting strategy, produce "
        "conformal forecast intervals, and diagnose "
        "their historical coverage using only calibration "
        "errors observable before each forecast origin."
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

        coverage = _validate_coverage(
            kwargs.get(
                "coverage",
                0.90,
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

        minimum_calibration_observations = (
            _validate_positive_integer(
                kwargs.get(
                    "minimum_calibration_observations",
                    3,
                ),
                "minimum_calibration_observations",
            )
        )

        calibration_tolerance = (
            _validate_calibration_tolerance(
                kwargs.get(
                    "calibration_tolerance",
                    0.10,
                )
            )
        )

        forecast_kwargs = {
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
            "horizon": horizon,
            "metric": metric,
            "coverage": coverage,
            "stride": stride,
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
        }

        if (
            kwargs.get(
                "initial_train_size"
            )
            is not None
        ):
            forecast_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        forecast = (
            AutomaticStrategyForecastWithIntervals()
            .execute(
                **forecast_kwargs
            )
        )

        selected_strategy = (
            forecast[
                "selected_strategy"
            ]
        )

        if (
            selected_strategy
            == "single_model"
        ):
            selected_model = (
                forecast[
                    "selected_model"
                ]
            )

            parameters = dict(
                forecast[
                    "selected_model_parameters"
                ]
                or {}
            )

            historical = (
                RollingOriginForecastBacktest()
                .execute(
                    time=list(
                        time
                    ),
                    values=(
                        values.tolist()
                    ),
                    model=selected_model,
                    initial_train_size=(
                        forecast[
                            "initial_train_size"
                        ]
                    ),
                    horizon=horizon,
                    stride=stride,
                    **parameters,
                )
            )

            historical_records = list(
                historical[
                    "records"
                ]
            )

            validation_source = (
                "selected_single_model_backtest"
            )

        elif (
            selected_strategy
            == "adaptive_ensemble"
        ):
            parameters_by_model = {
                candidate[
                    "model"
                ]: dict(
                    candidate[
                        "model_parameters"
                    ]
                    or {}
                )
                for candidate
                in forecast[
                    "strategy_ranking"
                ]
                if (
                    candidate[
                        "strategy"
                    ]
                    == "single_model"
                )
            }

            historical = (
                AdaptiveForecastEnsembleBacktest()
                .execute(
                    time=list(
                        time
                    ),
                    values=(
                        values.tolist()
                    ),
                    models=forecast[
                        "models"
                    ],
                    parameters_by_model=(
                        parameters_by_model
                    ),
                    initial_train_size=(
                        forecast[
                            "initial_train_size"
                        ]
                    ),
                    horizon=horizon,
                    stride=stride,
                    metric=metric,
                    minimum_weight_observations=(
                        minimum_weight_observations
                    ),
                )
            )

            historical_records = list(
                historical[
                    "records"
                ]
            )

            validation_source = (
                "adaptive_ensemble_backtest"
            )

        else:
            raise InvalidAnalysisInputError(
                "Unsupported selected forecast "
                f"strategy: '{selected_strategy}'."
            )

        calibration = (
            _sequential_interval_calibration(
                historical_records,
                horizon=horizon,
                coverage=coverage,
                minimum_calibration_observations=(
                    minimum_calibration_observations
                ),
            )
        )

        empirical_coverage = float(
            calibration[
                "empirical_coverage"
            ]
        )

        status = _calibration_status(
            empirical_coverage=(
                empirical_coverage
            ),
            nominal_coverage=coverage,
            tolerance=(
                calibration_tolerance
            ),
        )

        return {
            "analysis_type": (
                "automatic_strategy_forecast_reliability"
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
            "nominal_coverage": (
                coverage
            ),
            "empirical_coverage": (
                empirical_coverage
            ),
            "coverage_gap": (
                calibration[
                    "coverage_gap"
                ]
            ),
            "absolute_coverage_error": (
                calibration[
                    "absolute_coverage_error"
                ]
            ),
            "calibration_tolerance": (
                calibration_tolerance
            ),
            "calibration_status": (
                status
            ),
            "reliable_within_tolerance": (
                status
                == "well_calibrated"
            ),
            "selected_strategy": (
                selected_strategy
            ),
            "selected_strategy_name": (
                forecast[
                    "selected_strategy_name"
                ]
            ),
            "selected_model": (
                forecast[
                    "selected_model"
                ]
            ),
            "selected_model_parameters": (
                forecast[
                    "selected_model_parameters"
                ]
            ),
            "strategy_selection_score": (
                forecast[
                    "strategy_selection_score"
                ]
            ),
            "strategy_selection_metrics": (
                forecast[
                    "strategy_selection_metrics"
                ]
            ),
            "strategy_ranking": (
                forecast[
                    "strategy_ranking"
                ]
            ),
            "strategies_evaluated": (
                forecast[
                    "strategies_evaluated"
                ]
            ),
            "models_evaluated": (
                forecast[
                    "models_evaluated"
                ]
            ),
            "models": (
                forecast[
                    "models"
                ]
            ),
            "initial_train_size": (
                forecast[
                    "initial_train_size"
                ]
            ),
            "stride": stride,
            "minimum_weight_observations": (
                minimum_weight_observations
            ),
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
            "validation_source": (
                validation_source
            ),
            "validation_observations": (
                calibration[
                    "validation_observations"
                ]
            ),
            "covered_observations": (
                calibration[
                    "covered_observations"
                ]
            ),
            "average_interval_width": (
                calibration[
                    "average_interval_width"
                ]
            ),
            "median_interval_width": (
                calibration[
                    "median_interval_width"
                ]
            ),
            "per_horizon_calibration": (
                calibration[
                    "per_horizon"
                ]
            ),
            "historical_calibration_records": (
                calibration[
                    "records"
                ]
            ),
            "forecast_values": (
                forecast[
                    "forecast_values"
                ]
            ),
            "forecast_timestamps": (
                forecast[
                    "forecast_timestamps"
                ]
            ),
            "intervals": (
                forecast[
                    "intervals"
                ]
            ),
            "forecasts": (
                forecast[
                    "forecasts"
                ]
            ),
            "ensemble_weights": (
                forecast[
                    "ensemble_weights"
                ]
            ),
            "interval_calibration_look_ahead_leakage": False,
            "finite_sample_correction": True,
            "post_selection_diagnostic": True,
            "selection_adjusted_coverage_guarantee": False,
            "iid_coverage_guarantee_claimed": False,
            "diagnostic_note": (
                "At each historical forecast origin, "
                "the conformal radius uses only errors "
                "whose targets were already observed. "
                "However, the winning forecast strategy "
                "was selected using the available historical "
                "sample, so this remains a post-selection "
                "diagnostic rather than an independent "
                "selection-adjusted coverage guarantee."
            ),
        }
