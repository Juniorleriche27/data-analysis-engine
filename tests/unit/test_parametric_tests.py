import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


# ============================================================
# ONE-SAMPLE
# ============================================================

def test_one_sample_t_test_null_mean():
    result = DataAnalysisEngine().analyze(
        "one_sample_mean_t_test",
        values=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        hypothesized_mean=3.0,
    ).result

    assert result[
        "sample_mean"
    ] == pytest.approx(
        3.0
    )

    assert result[
        "t_statistic"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )


def test_one_sample_t_test_detects_positive_mean():
    result = DataAnalysisEngine().analyze(
        "one_sample_mean_t_test",
        values=[
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        hypothesized_mean=0.0,
        alternative="greater",
    ).result

    assert result[
        "t_statistic"
    ] > 0.0

    assert result[
        "p_value"
    ] < 0.05


def test_one_sample_t_test_tracks_missing_values():
    result = DataAnalysisEngine().analyze(
        "one_sample_mean_t_test",
        values=[
            1.0,
            None,
            2.0,
            np.nan,
            3.0,
        ],
        hypothesized_mean=2.0,
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "sample_size"
    ] == 3

    assert result[
        "missing_count"
    ] == 2


def test_one_sample_t_test_requires_two_observations():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "one_sample_mean_t_test",
            values=[
                1.0,
            ],
        )


# ============================================================
# WELCH
# ============================================================

def test_welch_t_test_equal_means_has_p_one():
    result = DataAnalysisEngine().analyze(
        "welch_mean_difference_t_test",
        data={
            "a": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "b": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        },
    ).result

    assert result[
        "observed_mean_difference"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "t_statistic"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )


def test_welch_t_test_detects_large_difference():
    result = DataAnalysisEngine().analyze(
        "welch_mean_difference_t_test",
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
    ).result

    assert result[
        "observed_mean_difference"
    ] < 0.0

    assert result[
        "p_value"
    ] < 0.05


def test_welch_t_test_computes_fractional_degrees_of_freedom():
    result = DataAnalysisEngine().analyze(
        "welch_mean_difference_t_test",
        data={
            "a": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "b": [
                2.0,
                10.0,
                20.0,
                30.0,
            ],
        },
    ).result

    assert result[
        "degrees_of_freedom"
    ] > 0.0

    assert np.isfinite(
        result[
            "degrees_of_freedom"
        ]
    )


def test_welch_t_test_tracks_independent_missing_values():
    result = DataAnalysisEngine().analyze(
        "welch_mean_difference_t_test",
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


# ============================================================
# PAIRED
# ============================================================

def test_paired_t_test_identical_pairs_has_p_one():
    result = DataAnalysisEngine().analyze(
        "paired_mean_difference_t_test",
        data={
            "before": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "after": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        },
    ).result

    assert result[
        "observed_mean_difference"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )


def test_paired_t_test_detects_consistent_positive_difference():
    result = DataAnalysisEngine().analyze(
        "paired_mean_difference_t_test",
        data={
            "before": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],
            "after": [
                5.0,
                15.0,
                25.0,
                35.0,
            ],
        },
        alternative="greater",
    ).result

    assert result[
        "observed_mean_difference"
    ] == pytest.approx(
        5.0
    )

    assert result[
        "t_statistic_infinite"
    ] is True

    assert result[
        "p_value"
    ] == pytest.approx(
        0.0
    )


def test_paired_t_test_uses_pairwise_missing_exclusion():
    result = DataAnalysisEngine().analyze(
        "paired_mean_difference_t_test",
        data={
            "before": [
                10.0,
                None,
                30.0,
                40.0,
            ],
            "after": [
                8.0,
                18.0,
                np.nan,
                35.0,
            ],
        },
    ).result

    assert result[
        "total_pair_count"
    ] == 4

    assert result[
        "complete_pair_count"
    ] == 2

    assert result[
        "missing_pair_count"
    ] == 2


def test_paired_t_test_rejects_unequal_lengths():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "paired_mean_difference_t_test",
            data={
                "before": [
                    1.0,
                    2.0,
                    3.0,
                ],
                "after": [
                    1.0,
                    2.0,
                ],
            },
        )
