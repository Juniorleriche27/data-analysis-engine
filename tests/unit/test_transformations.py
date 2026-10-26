import math

import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_log_transformation():
    result = DataAnalysisEngine().analyze(
        "numerical_transformation",
        values=[
            1.0,
            math.e,
            math.e ** 2,
        ],
        strategy="log",
    ).result

    assert result[
        "values"
    ] == pytest.approx(
        [
            0.0,
            1.0,
            2.0,
        ]
    )


def test_log1p_transformation():
    result = DataAnalysisEngine().analyze(
        "numerical_transformation",
        values=[
            0.0,
            1.0,
            3.0,
        ],
        strategy="log1p",
    ).result

    assert result[
        "values"
    ] == pytest.approx(
        [
            0.0,
            math.log(2.0),
            math.log(4.0),
        ]
    )


def test_sqrt_transformation():
    result = DataAnalysisEngine().analyze(
        "numerical_transformation",
        values=[
            0.0,
            4.0,
            9.0,
        ],
        strategy="sqrt",
    ).result

    assert result[
        "values"
    ] == pytest.approx(
        [
            0.0,
            2.0,
            3.0,
        ]
    )


def test_signed_log1p_supports_negative_values():
    result = DataAnalysisEngine().analyze(
        "numerical_transformation",
        values=[
            -3.0,
            0.0,
            3.0,
        ],
        strategy="signed_log1p",
    ).result

    expected = math.log(
        4.0
    )

    assert result[
        "values"
    ] == pytest.approx(
        [
            -expected,
            0.0,
            expected,
        ]
    )


def test_transformation_preserves_missing_positions():
    result = DataAnalysisEngine().analyze(
        "numerical_transformation",
        values=[
            0.0,
            None,
            3.0,
            np.nan,
        ],
        strategy="log1p",
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


def test_log_rejects_zero_and_negative_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "numerical_transformation",
            values=[
                1.0,
                0.0,
                2.0,
            ],
            strategy="log",
        )


def test_sqrt_rejects_negative_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "numerical_transformation",
            values=[
                1.0,
                -1.0,
                2.0,
            ],
            strategy="sqrt",
        )
