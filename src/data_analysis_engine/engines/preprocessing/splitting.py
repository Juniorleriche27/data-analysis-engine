from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_ALLOWED_STRATEGIES = {
    "random",
    "temporal",
}


def _validate_strategy(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'strategy' must be a string."
        )

    strategy = value.strip().lower()

    if strategy not in _ALLOWED_STRATEGIES:
        raise InvalidAnalysisInputError(
            "'strategy' must be one of: "
            "random, temporal."
        )

    return strategy


def _validate_test_size(
    value: Any,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise InvalidAnalysisInputError(
            "'test_size' must be numeric."
        )

    test_size = float(value)

    if (
        not np.isfinite(test_size)
        or test_size <= 0.0
        or test_size >= 1.0
    ):
        raise InvalidAnalysisInputError(
            "'test_size' must satisfy "
            "0 < test_size < 1."
        )

    return test_size


def _validate_random_state(
    value: Any,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise InvalidAnalysisInputError(
            "'random_state' must be an integer."
        )

    return int(value)


def _python_scalar(
    value: Any,
) -> Any:
    if pd.isna(value):
        return None

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def _serialize_frame(
    frame: pd.DataFrame,
) -> dict[str, list[Any]]:
    return {
        str(column): [
            _python_scalar(value)
            for value
            in frame[column]
        ]
        for column
        in frame.columns
    }


class TrainTestSplit(
    AnalysisMethod
):
    name = "train_test_split"
    category = "preprocessing"
    version = "1.0"

    description = (
        "Split a dataset into deterministic training and "
        "test partitions using reproducible random splitting "
        "or chronological leakage-safe temporal splitting."
    )

    supported_roles = (
        "time",
        "measure",
        "dimension",
        "text",
    )

    minimum_compatible_columns = 1

    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = ()

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = kwargs.get(
            "data"
        )

        if not isinstance(
            data,
            dict,
        ):
            raise InvalidAnalysisInputError(
                "'data' must be a dictionary "
                "of column names to values."
            )

        if not data:
            raise InvalidAnalysisInputError(
                "Train/test splitting requires "
                "at least one column."
            )

        strategy = _validate_strategy(
            kwargs.get(
                "strategy",
                "random",
            )
        )

        test_size = _validate_test_size(
            kwargs.get(
                "test_size",
                0.20,
            )
        )

        random_state = _validate_random_state(
            kwargs.get(
                "random_state",
                42,
            )
        )

        try:
            frame = pd.DataFrame(
                data
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "All columns must form a rectangular "
                "dataframe."
            ) from exc

        row_count = int(
            len(frame)
        )

        if row_count < 2:
            raise InvalidAnalysisInputError(
                "Train/test splitting requires "
                "at least two rows."
            )

        test_count = int(
            np.ceil(
                row_count
                * test_size
            )
        )

        test_count = max(
            1,
            min(
                test_count,
                row_count - 1,
            ),
        )

        train_count = int(
            row_count
            - test_count
        )

        if strategy == "random":
            generator = (
                np.random.default_rng(
                    random_state
                )
            )

            permutation = generator.permutation(
                row_count
            )

            test_indices = np.sort(
                permutation[
                    :test_count
                ]
            )

            train_indices = np.sort(
                permutation[
                    test_count:
                ]
            )

            temporal_leakage_protection = False
            look_ahead_leakage = None
            time_column = None
            train_end_timestamp = None
            test_start_timestamp = None

        else:
            time_column = kwargs.get(
                "time_column"
            )

            if not isinstance(
                time_column,
                str,
            ) or not time_column.strip():
                raise InvalidAnalysisInputError(
                    "'time_column' is required when "
                    "strategy='temporal'."
                )

            if time_column not in frame.columns:
                raise InvalidAnalysisInputError(
                    f"Time column '{time_column}' "
                    "was not found in the dataset."
                )

            time_values = frame[
                time_column
            ]

            if bool(
                time_values.isna().any()
            ):
                raise InvalidAnalysisInputError(
                    "Temporal splitting does not allow "
                    "missing timestamps."
                )

            parsed = pd.to_datetime(
                time_values,
                errors="coerce",
                utc=True,
            )

            if bool(
                parsed.isna().any()
            ):
                raise InvalidAnalysisInputError(
                    "Temporal splitting requires all "
                    "timestamps to be parseable."
                )

            if bool(
                parsed.duplicated(
                    keep=False
                ).any()
            ):
                raise InvalidAnalysisInputError(
                    "Temporal splitting requires unique "
                    "timestamps to guarantee an unambiguous "
                    "leakage-safe boundary."
                )

            timestamp_ns = np.asarray(
                [
                    timestamp.value
                    for timestamp
                    in parsed
                ],
                dtype=np.int64,
            )

            order = np.argsort(
                timestamp_ns,
                kind="stable",
            )

            train_indices = order[
                :train_count
            ]

            test_indices = order[
                train_count:
            ]

            train_times = timestamp_ns[
                train_indices
            ]

            test_times = timestamp_ns[
                test_indices
            ]

            train_end_ns = int(
                np.max(
                    train_times
                )
            )

            test_start_ns = int(
                np.min(
                    test_times
                )
            )

            if train_end_ns >= test_start_ns:
                raise InvalidAnalysisInputError(
                    "Temporal split boundary is not "
                    "strictly leakage-safe."
                )

            temporal_leakage_protection = True
            look_ahead_leakage = False

            train_end_timestamp = (
                pd.Timestamp(
                    train_end_ns,
                    unit="ns",
                    tz="UTC",
                ).isoformat()
            )

            test_start_timestamp = (
                pd.Timestamp(
                    test_start_ns,
                    unit="ns",
                    tz="UTC",
                ).isoformat()
            )

        train_frame = frame.iloc[
            train_indices
        ].copy()

        test_frame = frame.iloc[
            test_indices
        ].copy()

        train_index_list = [
            int(index)
            for index
            in train_indices
        ]

        test_index_list = [
            int(index)
            for index
            in test_indices
        ]

        overlap = set(
            train_index_list
        ).intersection(
            test_index_list
        )

        if overlap:
            raise InvalidAnalysisInputError(
                "Train/test partitions overlap."
            )

        covered = set(
            train_index_list
        ).union(
            test_index_list
        )

        if len(covered) != row_count:
            raise InvalidAnalysisInputError(
                "Train/test partitions do not cover "
                "the complete dataset."
            )

        return {
            "transformation_type": (
                "train_test_split"
            ),
            "strategy": strategy,
            "row_count": row_count,
            "train_count": train_count,
            "test_count": test_count,
            "requested_test_size": (
                test_size
            ),
            "actual_test_fraction": float(
                test_count
                / row_count
            ),
            "random_state": (
                random_state
                if strategy == "random"
                else None
            ),
            "time_column": (
                time_column
            ),
            "train_indices": (
                train_index_list
            ),
            "test_indices": (
                test_index_list
            ),
            "train_data": (
                _serialize_frame(
                    train_frame
                )
            ),
            "test_data": (
                _serialize_frame(
                    test_frame
                )
            ),
            "partitions_disjoint": True,
            "complete_partition": True,
            "temporal_leakage_protection": (
                temporal_leakage_protection
            ),
            "look_ahead_leakage": (
                look_ahead_leakage
            ),
            "train_end_timestamp": (
                train_end_timestamp
            ),
            "test_start_timestamp": (
                test_start_timestamp
            ),
        }
