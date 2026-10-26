import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_random_split_is_reproducible():
    engine = DataAnalysisEngine()

    parameters = {
        "data": {
            "x": list(
                range(10)
            )
        },
        "strategy": "random",
        "test_size": 0.2,
        "random_state": 42,
    }

    first = engine.analyze(
        "train_test_split",
        **parameters,
    ).result

    second = engine.analyze(
        "train_test_split",
        **parameters,
    ).result

    assert (
        first["train_indices"]
        == second["train_indices"]
    )

    assert (
        first["test_indices"]
        == second["test_indices"]
    )


def test_random_split_has_disjoint_complete_partitions():
    result = DataAnalysisEngine().analyze(
        "train_test_split",
        data={
            "x": list(
                range(10)
            )
        },
        strategy="random",
        test_size=0.3,
        random_state=7,
    ).result

    train = set(
        result[
            "train_indices"
        ]
    )

    test = set(
        result[
            "test_indices"
        ]
    )

    assert train.isdisjoint(
        test
    )

    assert train | test == set(
        range(10)
    )

    assert result[
        "train_count"
    ] == 7

    assert result[
        "test_count"
    ] == 3


def test_temporal_split_keeps_future_in_test():
    result = DataAnalysisEngine().analyze(
        "train_test_split",
        data={
            "timestamp": [
                "2026-01-03",
                "2026-01-01",
                "2026-01-05",
                "2026-01-02",
                "2026-01-04",
            ],
            "value": [
                30,
                10,
                50,
                20,
                40,
            ],
        },
        strategy="temporal",
        test_size=0.4,
        time_column="timestamp",
    ).result

    assert result[
        "train_indices"
    ] == [
        1,
        3,
        0,
    ]

    assert result[
        "test_indices"
    ] == [
        4,
        2,
    ]

    assert (
        result[
            "temporal_leakage_protection"
        ]
        is True
    )

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )

    assert (
        result[
            "train_end_timestamp"
        ]
        < result[
            "test_start_timestamp"
        ]
    )


def test_temporal_split_preserves_all_columns():
    result = DataAnalysisEngine().analyze(
        "train_test_split",
        data={
            "timestamp": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "value": [
                10,
                20,
                30,
                40,
            ],
            "category": [
                "A",
                "B",
                "C",
                "D",
            ],
        },
        strategy="temporal",
        test_size=0.5,
        time_column="timestamp",
    ).result

    assert set(
        result[
            "train_data"
        ]
    ) == {
        "timestamp",
        "value",
        "category",
    }

    assert set(
        result[
            "test_data"
        ]
    ) == {
        "timestamp",
        "value",
        "category",
    }


def test_temporal_split_rejects_duplicate_timestamps():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "train_test_split",
            data={
                "timestamp": [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                ],
                "value": [
                    1,
                    2,
                    3,
                ],
            },
            strategy="temporal",
            time_column="timestamp",
        )


def test_temporal_split_requires_time_column():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "train_test_split",
            data={
                "x": [
                    1,
                    2,
                    3,
                ]
            },
            strategy="temporal",
        )


def test_split_rejects_invalid_test_size():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "train_test_split",
            data={
                "x": [
                    1,
                    2,
                    3,
                ]
            },
            test_size=1.0,
        )
