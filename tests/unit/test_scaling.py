import math

import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_standard_scaling_centers_and_scales():
    result = DataAnalysisEngine().analyze(
        "feature_scaling",
        values=[
            1.0,
            2.0,
            3.0,
        ],
        strategy="standard",
    ).result

    values = result[
        "values"
    ]

    assert sum(
        values
    ) / len(
        values
    ) == pytest.approx(
        0.0
    )

    variance = sum(
        value ** 2
        for value
        in values
    ) / len(
        values
    )

    assert variance == pytest.approx(
        1.0
    )


def test_standard_scaling_constant_column_returns_zero():
    result = DataAnalysisEngine().analyze(
        "feature_scaling",
        values=[
            5.0,
            5.0,
            5.0,
        ],
        strategy="standard",
    ).result

    assert result[
        "is_constant"
    ] is True

    assert result[
        "values"
    ] == [
        0.0,
        0.0,
        0.0,
    ]


def test_minmax_scaling_default_range():
    result = DataAnalysisEngine().analyze(
        "feature_scaling",
        values=[
            10.0,
            20.0,
            30.0,
        ],
        strategy="minmax",
    ).result

    assert result[
        "values"
    ] == pytest.approx(
        [
            0.0,
            0.5,
            1.0,
        ]
    )


def test_minmax_scaling_custom_range():
    result = DataAnalysisEngine().analyze(
        "feature_scaling",
        values=[
            10.0,
            20.0,
            30.0,
        ],
        strategy="minmax",
        feature_range=[
            -1.0,
            1.0,
        ],
    ).result

    assert result[
        "values"
    ] == pytest.approx(
        [
            -1.0,
            0.0,
            1.0,
        ]
    )


def test_scaling_preserves_missing_positions():
    result = DataAnalysisEngine().analyze(
        "feature_scaling",
        values=[
            1.0,
            None,
            3.0,
            np.nan,
        ],
        strategy="minmax",
    ).result

    assert result[
        "missing_count"
    ] == 2

    assert result[
        "missing_positions"
    ] == [
        1,
        3,
    ]

    assert result[
        "values"
    ][1] is None

    assert result[
        "values"
    ][3] is None

    assert result[
        "values"
    ][0] == pytest.approx(
        0.0
    )

    assert result[
        "values"
    ][2] == pytest.approx(
        1.0
    )


def test_scaling_rejects_infinite_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "feature_scaling",
            values=[
                1.0,
                math.inf,
                3.0,
            ],
        )


def test_scaling_rejects_text_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "feature_scaling",
            values=[
                "1",
                "2",
                "3",
            ],
            strategy="standard",
        )
