import math

import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_rolling_volatility_from_returns():
    result = DataAnalysisEngine().analyze(
        "rolling_volatility",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
        ],
        values=[
            0.10,
            -0.10,
            0.10,
            -0.10,
        ],
        input_kind="returns",
        window=2,
    )

    records = result.result[
        "records"
    ]

    assert records[0][
        "volatility"
    ] == pytest.approx(
        math.sqrt(0.02)
    )


def test_rolling_volatility_can_convert_prices():
    result = DataAnalysisEngine().analyze(
        "rolling_volatility",
        time=[
            "t1",
            "t2",
            "t3",
        ],
        values=[
            100,
            110,
            121,
        ],
        input_kind="prices",
        return_type="simple",
        window=2,
    )

    assert (
        result.result[
            "transformation"
        ]
        == "simple_returns"
    )

    assert result.result[
        "records"
    ][0]["volatility"] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_rolling_volatility_annualization():
    result = DataAnalysisEngine().analyze(
        "rolling_volatility",
        time=[
            "t1",
            "t2",
            "t3",
        ],
        values=[
            0.01,
            -0.01,
            0.01,
        ],
        input_kind="returns",
        window=2,
        annualization_factor=252,
    )

    record = result.result[
        "records"
    ][0]

    assert record[
        "volatility"
    ] == pytest.approx(
        record[
            "raw_volatility"
        ]
        * math.sqrt(252)
    )


def test_rolling_volatility_rejects_invalid_window():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "rolling_volatility",
            time=[
                "t1",
                "t2",
            ],
            values=[
                0.01,
                0.02,
            ],
            input_kind="returns",
            window=3,
        )


def test_ewma_zero_returns_produce_zero_volatility():
    result = DataAnalysisEngine().analyze(
        "ewma_volatility",
        time=[
            "t1",
            "t2",
            "t3",
        ],
        values=[
            0,
            0,
            0,
        ],
        input_kind="returns",
    )

    assert all(
        record[
            "volatility"
        ] == pytest.approx(0.0)
        for record
        in result.result[
            "records"
        ]
    )


def test_ewma_recursive_variance():
    result = DataAnalysisEngine().analyze(
        "ewma_volatility",
        time=[
            "t1",
            "t2",
        ],
        values=[
            0.10,
            0.20,
        ],
        input_kind="returns",
        decay=0.5,
    )

    records = result.result[
        "records"
    ]

    assert records[0][
        "variance"
    ] == pytest.approx(
        0.01
    )

    assert records[1][
        "variance"
    ] == pytest.approx(
        0.025
    )


def test_ewma_rejects_invalid_decay():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "ewma_volatility",
            time=[
                "t1",
                "t2",
            ],
            values=[
                0.01,
                0.02,
            ],
            input_kind="returns",
            decay=1.0,
        )
