import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_winsorization_clips_both_tails():
    result = DataAnalysisEngine().analyze(
        "winsorization",
        values=[
            0.0,
            10.0,
            20.0,
            30.0,
            100.0,
        ],
        lower_quantile=0.20,
        upper_quantile=0.80,
    ).result

    assert result[
        "lower_clipped_count"
    ] == 1

    assert result[
        "upper_clipped_count"
    ] == 1

    assert result[
        "clipped_count"
    ] == 2

    assert result[
        "clipped_positions"
    ] == [
        0,
        4,
    ]


def test_winsorization_computes_expected_bounds():
    result = DataAnalysisEngine().analyze(
        "winsorization",
        values=[
            0.0,
            10.0,
            20.0,
            30.0,
            100.0,
        ],
        lower_quantile=0.20,
        upper_quantile=0.80,
    ).result

    assert result[
        "lower_bound"
    ] == pytest.approx(
        8.0
    )

    assert result[
        "upper_bound"
    ] == pytest.approx(
        44.0
    )

    assert result[
        "values"
    ] == pytest.approx(
        [
            8.0,
            10.0,
            20.0,
            30.0,
            44.0,
        ]
    )


def test_winsorization_preserves_missing_positions():
    result = DataAnalysisEngine().analyze(
        "winsorization",
        values=[
            0.0,
            None,
            10.0,
            20.0,
            100.0,
        ],
        lower_quantile=0.25,
        upper_quantile=0.75,
    ).result

    assert result[
        "missing_count"
    ] == 1

    assert result[
        "values"
    ][1] is None

    assert 1 not in result[
        "clipped_positions"
    ]


def test_winsorization_constant_column_is_noop():
    result = DataAnalysisEngine().analyze(
        "winsorization",
        values=[
            5.0,
            5.0,
            5.0,
        ],
        lower_quantile=0.10,
        upper_quantile=0.90,
    ).result

    assert result[
        "changed"
    ] is False

    assert result[
        "clipped_count"
    ] == 0

    assert result[
        "values"
    ] == pytest.approx(
        [
            5.0,
            5.0,
            5.0,
        ]
    )


def test_winsorization_rejects_invalid_quantiles():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "winsorization",
            values=[
                1.0,
                2.0,
                3.0,
            ],
            lower_quantile=0.9,
            upper_quantile=0.1,
        )


def test_winsorization_rejects_non_finite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "winsorization",
            values=[
                1.0,
                np.inf,
                3.0,
            ],
        )


def test_winsorization_rejects_text_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "winsorization",
            values=[
                "1",
                "2",
                "3",
            ],
        )
