import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_one_sample_proportion_null_case():
    result = DataAnalysisEngine().analyze(
        "one_sample_proportion_z_test",
        values=[
            1,
            0,
            1,
            0,
        ],
        hypothesized_proportion=0.5,
    ).result

    assert result[
        "observed_proportion"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "z_statistic"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )


def test_one_sample_proportion_supports_categorical_success():
    result = DataAnalysisEngine().analyze(
        "one_sample_proportion_z_test",
        values=[
            "yes",
            "yes",
            "yes",
            "yes",
            "no",
        ],
        success_value="yes",
        hypothesized_proportion=0.5,
        alternative="greater",
    ).result

    assert result[
        "success_count"
    ] == 4

    assert result[
        "observed_proportion"
    ] == pytest.approx(
        0.8
    )

    assert result[
        "z_statistic"
    ] > 0.0


def test_one_sample_proportion_tracks_missing_values():
    result = DataAnalysisEngine().analyze(
        "one_sample_proportion_z_test",
        values=[
            1,
            None,
            0,
            np.nan,
            1,
        ],
        hypothesized_proportion=0.5,
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


def test_one_sample_proportion_rejects_boundary_null():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "one_sample_proportion_z_test",
            values=[
                1,
                0,
                1,
            ],
            hypothesized_proportion=1.0,
        )


def test_two_sample_proportion_equal_samples():
    result = DataAnalysisEngine().analyze(
        "two_sample_proportion_z_test",
        data={
            "a": [
                1,
                0,
                1,
                0,
            ],
            "b": [
                1,
                0,
                1,
                0,
            ],
        },
    ).result

    assert result[
        "observed_proportion_difference"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )


def test_two_sample_proportion_detects_difference():
    result = DataAnalysisEngine().analyze(
        "two_sample_proportion_z_test",
        data={
            "control": [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "treatment": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
            ],
        },
    ).result

    assert result[
        "first_proportion"
    ] == pytest.approx(
        0.1
    )

    assert result[
        "second_proportion"
    ] == pytest.approx(
        0.9
    )

    assert result[
        "p_value"
    ] < 0.01


def test_two_sample_proportion_tracks_missing_values():
    result = DataAnalysisEngine().analyze(
        "two_sample_proportion_z_test",
        data={
            "a": [
                1,
                None,
                0,
                1,
            ],
            "b": [
                0,
                1,
                np.nan,
                0,
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


def test_two_sample_proportion_requires_pair_selection_for_extra_columns():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "two_sample_proportion_z_test",
            data={
                "a": [
                    1,
                    0,
                ],
                "b": [
                    0,
                    1,
                ],
                "c": [
                    1,
                    1,
                ],
            },
        )
