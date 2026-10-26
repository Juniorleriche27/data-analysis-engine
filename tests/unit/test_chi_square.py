import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_chi_square_independent_table_has_zero_statistic():
    result = DataAnalysisEngine().analyze(
        "chi_square_independence_test",
        data={
            "group": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "outcome": [
                "yes",
                "yes",
                "no",
                "no",
                "yes",
                "yes",
                "no",
                "no",
            ],
        },
    ).result

    assert result[
        "chi_square_statistic"
    ] == pytest.approx(
        0.0
    )

    assert result[
        "p_value"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "cramers_v"
    ] == pytest.approx(
        0.0
    )


def test_chi_square_detects_strong_association():
    result = DataAnalysisEngine().analyze(
        "chi_square_independence_test",
        data={
            "group": (
                ["A"] * 20
                + ["B"] * 20
            ),
            "outcome": (
                ["yes"] * 18
                + ["no"] * 2
                + ["yes"] * 2
                + ["no"] * 18
            ),
        },
    ).result

    assert result[
        "chi_square_statistic"
    ] > 0.0

    assert result[
        "p_value"
    ] < 0.001

    assert result[
        "cramers_v"
    ] > 0.5


def test_chi_square_uses_pairwise_complete_rows():
    result = DataAnalysisEngine().analyze(
        "chi_square_independence_test",
        data={
            "group": [
                "A",
                "A",
                None,
                "B",
                "B",
            ],
            "outcome": [
                "yes",
                "no",
                "yes",
                None,
                "no",
            ],
        },
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "complete_pair_count"
    ] == 3

    assert result[
        "missing_pair_count"
    ] == 2


def test_chi_square_rejects_constant_category():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "chi_square_independence_test",
            data={
                "group": [
                    "A",
                    "A",
                    "A",
                    "A",
                ],
                "outcome": [
                    "yes",
                    "no",
                    "yes",
                    "no",
                ],
            },
        )
