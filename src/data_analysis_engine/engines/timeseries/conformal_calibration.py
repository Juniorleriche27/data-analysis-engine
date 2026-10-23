from typing import Any

import numpy as np

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.timeseries.conformal_intervals import (
    _finite_sample_conformal_radius,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
    SUPPORTED_MODELS,
    _validate_positive_integer,
)
from data_analysis_engine.engines.timeseries.forecast_intervals import (
    _validate_coverage,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    _prepare_regular_series,
    _validate_horizon,
)


class ConformalIntervalCalibration(
    AnalysisMethod
):
    name = "conformal_interval_calibration"
    category = "timeseries"
    version = "1.0"

    description = (
        "Evaluate leakage-safe historical coverage of "
        "finite-sample conformal-style forecast intervals."
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

        if model not in SUPPORTED_MODELS:
            raise InvalidAnalysisInputError(
                "'model' must be one of: "
                + ", ".join(
                    SUPPORTED_MODELS
                )
                + "."
            )

        horizon = _validate_horizon(
            kwargs.get(
                "horizon",
                1,
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

        minimum_calibration_observations = (
            _validate_positive_integer(
                kwargs.get(
                    "minimum_calibration_observations",
                    3,
                ),
                "minimum_calibration_observations",
            )
        )

        backtest_kwargs = {
            "time": list(time),
            "values": values.tolist(),
            "model": model,
            "horizon": horizon,
            "stride": stride,
        }

        if (
            kwargs.get(
                "initial_train_size"
            )
            is not None
        ):
            backtest_kwargs[
                "initial_train_size"
            ] = kwargs[
                "initial_train_size"
            ]

        if model == "autoregressive":
            backtest_kwargs[
                "p"
            ] = kwargs.get(
                "p",
                1,
            )

        elif model == "seasonal_naive":
            backtest_kwargs[
                "season_length"
            ] = kwargs.get(
                "season_length",
                7,
            )

        elif (
            model
            == "simple_exponential_smoothing"
        ):
            backtest_kwargs[
                "alpha"
            ] = kwargs.get(
                "alpha",
                0.2,
            )

        elif model == "holt_linear_trend":
            backtest_kwargs[
                "alpha"
            ] = kwargs.get(
                "alpha",
                0.2,
            )

            backtest_kwargs[
                "beta"
            ] = kwargs.get(
                "beta",
                0.1,
            )

        elif model == "holt_winters_additive":
            backtest_kwargs[
                "alpha"
            ] = kwargs.get(
                "alpha",
                0.2,
            )

            backtest_kwargs[
                "beta"
            ] = kwargs.get(
                "beta",
                0.1,
            )

            backtest_kwargs[
                "gamma"
            ] = kwargs.get(
                "gamma",
                0.1,
            )

            backtest_kwargs[
                "season_length"
            ] = kwargs.get(
                "season_length",
                7,
            )

        backtest = (
            RollingOriginForecastBacktest()
            .execute(
                **backtest_kwargs
            )
        )

        records = backtest[
            "records"
        ]

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

            available_history = [
                historical
                for historical
                in records_by_step[
                    step
                ]
                if int(
                    historical[
                        "target_index"
                    ]
                )
                < training_size
            ]

            if (
                len(
                    available_history
                )
                < minimum_calibration_observations
            ):
                continue

            absolute_errors = np.asarray(
                [
                    abs(
                        float(
                            historical[
                                "error"
                            ]
                        )
                    )
                    for historical
                    in available_history
                ],
                dtype=float,
            )

            calibration = (
                _finite_sample_conformal_radius(
                    absolute_errors,
                    coverage=coverage,
                )
            )

            radius = float(
                calibration[
                    "radius"
                ]
            )

            point_forecast = float(
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
                point_forecast
                - radius
            )

            upper_bound = float(
                point_forecast
                + radius
            )

            covered = bool(
                lower_bound
                <= actual
                <= upper_bound
            )

            calibration_target_indices = [
                int(
                    historical[
                        "target_index"
                    ]
                )
                for historical
                in available_history
            ]

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
                    "point_forecast": (
                        point_forecast
                    ),
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
                        calibration[
                            "sample_size"
                        ]
                    ),
                    "conformal_rank": (
                        calibration[
                            "rank"
                        ]
                    ),
                    "empirical_quantile_level": (
                        calibration[
                            "empirical_quantile_level"
                        ]
                    ),
                    "calibration_max_target_index": (
                        max(
                            calibration_target_indices
                        )
                    ),
                }
            )

        if not validation_records:
            raise InvalidAnalysisInputError(
                "No conformal validation observations "
                "were available after the leakage-safe "
                "calibration warm-up. Increase the "
                "historical sample or lower "
                "'minimum_calibration_observations'."
            )

        covered_values = np.asarray(
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
                covered_values
            )
        )

        coverage_gap = float(
            empirical_coverage
            - coverage
        )

        per_horizon: list[
            dict[str, Any]
        ] = []

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
                        "average_radius": None,
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

            step_radii = np.asarray(
                [
                    record[
                        "radius"
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
                    "average_radius": float(
                        np.mean(
                            step_radii
                        )
                    ),
                }
            )

        return {
            "validation_type": (
                "leakage_safe_conformal_interval_calibration"
            ),
            "interval_type": (
                "symmetric_finite_sample_conformal"
            ),
            "model": model,
            "model_parameters": (
                backtest[
                    "model_parameters"
                ]
            ),
            "sample_size": len(
                values
            ),
            "nominal_coverage": (
                coverage
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
            "horizon": horizon,
            "stride": stride,
            "minimum_calibration_observations": (
                minimum_calibration_observations
            ),
            "initial_train_size": (
                backtest[
                    "initial_train_size"
                ]
            ),
            "backtest_origin_count": (
                backtest[
                    "origin_count"
                ]
            ),
            "validation_observations": (
                len(
                    validation_records
                )
            ),
            "covered_observations": int(
                np.sum(
                    covered_values
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
            "look_ahead_leakage": False,
            "finite_sample_correction": True,
            "nonconformity_score": (
                "absolute_forecast_error"
            ),
            "iid_coverage_guarantee_claimed": False,
            "calibration_rule": (
                "only_absolute_errors_with_target_index_"
                "before_current_training_origin"
            ),
            "frequency_seconds": (
                interval_ns
                / 1_000_000_000.0
            ),
            "per_horizon": (
                per_horizon
            ),
            "records": (
                validation_records
            ),
        }
