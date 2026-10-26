import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_permutation_detects_large_mean_difference():
    result = DataAnalysisEngine().analyze(
        "permutation_mean_difference_test",
        data={
            "control": [
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
            ],
            "treatment": [
                10.0,
                11.0,
                10.0,
                11.0,
                10.0,
            ],
        },
        n_resamples=3000,
        random_state=42,
    ).result

    assert result[
        "observed_mean_difference"
    ] < 0.0

    assert result[
        "p_value"
    ] < 0.05


def test_permutation_is_reproducible():
    engine = DataAnalysisEngine()

    parameters = {
        "data": {
            "a": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "b": [
                2.0,
                4.0,
                6.0,
                8.0,
            ],
        },
        "n_resamples": 1000,
        "random_state": 11,
    }

    first = engine.analyze(
        "permutation_mean_difference_test",
        **parameters,
    ).result

    second = engine.analyze(
        "permutation_mean_difference_test",
        **parameters,
    ).result

    assert first[
        "p_value"
    ] == second[
        "p_value"
    ]

    assert first[
        "extreme_count"
    ] == second[
        "extreme_count"
    ]


def test_permutation_supports_directional_alternative():
    result = DataAnalysisEngine().analyze(
        "permutation_mean_difference_test",
        data={
            "high": [
                8.0,
                9.0,
                10.0,
                11.0,
            ],
            "low": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        },
        alternative="greater",
        n_resamples=2000,
        random_state=5,
    ).result

    assert result[
        "observed_mean_difference"
    ] > 0.0

    assert result[
        "p_value"
    ] < 0.05


def test_permutation_preserves_independent_missing_counts():
    result = DataAnalysisEngine().analyze(
        "permutation_mean_difference_test",
        data={
            "a": [
                1.0,
                None,
                3.0,
                4.0,
            ],
            "b": [
                2.0,
                3.0,
                np.nan,
                5.0,
            ],
        },
        n_resamples=500,
    ).result

    assert result[
        "first_sample_size"
    ] == 3

    assert result[
        "second_sample_size"
    ] == 3

    assert result[
        "first_missing_count"
    ] == 1

    assert result[
        "second_missing_count"
    ] == 1


def test_permutation_requires_explicit_pair_when_more_than_two_columns():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "permutation_mean_difference_test",
            data={
                "a": [
                    1.0,
                    2.0,
                ],
                "b": [
                    3.0,
                    4.0,
                ],
                "c": [
                    5.0,
                    6.0,
                ],
            },
            n_resamples=500,
        )
