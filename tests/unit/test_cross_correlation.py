import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_cross_correlation_detects_x_leading_y():
    x = [
        1,
        4,
        2,
        8,
        3,
        7,
        5,
        9,
    ]

    y = [
        0,
        1,
        4,
        2,
        8,
        3,
        7,
        5,
    ]

    result = DataAnalysisEngine().analyze(
        "cross_correlation",
        x=x,
        y=y,
        max_lag=2,
    )

    strongest = result.result[
        "strongest_relationship"
    ]

    assert strongest[
        "lag"
    ] == 1

    assert strongest[
        "direction"
    ] == "x_leads_y"

    assert strongest[
        "coefficient"
    ] == pytest.approx(1.0)


def test_cross_correlation_zero_lag():
    result = DataAnalysisEngine().analyze(
        "cross_correlation",
        x=[1, 5, 2, 8, 3, 9],
        y=[2, 10, 4, 16, 6, 18],
        max_lag=0,
    )

    strongest = result.result[
        "strongest_relationship"
    ]

    assert strongest["lag"] == 0

    assert (
        strongest["direction"]
        == "simultaneous"
    )

    assert (
        strongest["coefficient"]
        == pytest.approx(1.0)
    )


def test_cross_correlation_produces_all_requested_lags():
    result = DataAnalysisEngine().analyze(
        "cross_correlation",
        x=[1, 4, 2, 8, 3, 7],
        y=[2, 1, 5, 3, 9, 4],
        max_lag=2,
    )

    lags = [
        item["lag"]
        for item in result.result[
            "correlations"
        ]
    ]

    assert lags == [
        -2,
        -1,
        0,
        1,
        2,
    ]


def test_cross_correlation_rejects_excessive_lag():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "cross_correlation",
            x=[1, 2, 3, 4],
            y=[4, 3, 2, 1],
            max_lag=3,
        )
