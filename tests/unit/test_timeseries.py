import math

import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_simple_returns():
    result = DataAnalysisEngine().analyze(
        "simple_returns",
        time=["t1", "t2", "t3"],
        values=[100, 110, 99],
    )

    assert result.result["returns"] == pytest.approx(
        [
            0.10,
            -0.10,
        ]
    )


def test_log_returns():
    result = DataAnalysisEngine().analyze(
        "log_returns",
        time=["t1", "t2"],
        values=[100, 110],
    )

    assert result.result["returns"][0] == pytest.approx(
        math.log(1.1)
    )


def test_log_returns_reject_non_positive_values():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "log_returns",
            time=["t1", "t2", "t3"],
            values=[100, 0, 110],
        )


def test_rolling_statistics():
    result = DataAnalysisEngine().analyze(
        "rolling_statistics",
        time=["t1", "t2", "t3", "t4"],
        values=[10, 20, 30, 40],
        window=2,
    )

    records = result.result["records"]

    assert len(records) == 3
    assert records[0]["mean"] == pytest.approx(15.0)
    assert records[1]["mean"] == pytest.approx(25.0)
    assert records[2]["mean"] == pytest.approx(35.0)


def test_autocorrelation():
    result = DataAnalysisEngine().analyze(
        "autocorrelation",
        time=["t1", "t2", "t3", "t4", "t5"],
        values=[1, 2, 3, 4, 5],
        lag=1,
    )

    assert result.result[
        "coefficient"
    ] == pytest.approx(1.0)


def test_lagged_series():
    result = DataAnalysisEngine().analyze(
        "lagged_series",
        time=["t1", "t2", "t3", "t4"],
        values=[10, 20, 30, 40],
        lag=1,
    )

    records = result.result["records"]

    assert result.result["paired_observations"] == 3

    assert records[0] == {
        "timestamp": "t2",
        "value": 20.0,
        "lagged_timestamp": "t1",
        "lagged_value": 10.0,
    }


def test_lagged_series_with_lag_two():
    result = DataAnalysisEngine().analyze(
        "lagged_series",
        time=["t1", "t2", "t3", "t4", "t5"],
        values=[10, 20, 30, 40, 50],
        lag=2,
    )

    assert (
        result.result["paired_observations"]
        == 3
    )

    assert (
        result.result["records"][0]["lagged_value"]
        == pytest.approx(10.0)
    )

    assert (
        result.result["records"][0]["value"]
        == pytest.approx(30.0)
    )


def test_lagged_series_rejects_invalid_lag():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "lagged_series",
            time=["t1", "t2", "t3"],
            values=[10, 20, 30],
            lag=3,
        )


def test_autocorrelation_function():
    result = DataAnalysisEngine().analyze(
        "autocorrelation_function",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
        ],
        values=[1, 2, 3, 4, 5, 6],
        max_lag=3,
    )

    acf = result.result[
        "autocorrelations"
    ]

    assert len(acf) == 3

    assert acf[0]["coefficient"] == pytest.approx(1.0)
    assert acf[1]["coefficient"] == pytest.approx(1.0)
    assert acf[2]["coefficient"] == pytest.approx(1.0)


def test_autocorrelation_function_rejects_excessive_lag():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "autocorrelation_function",
            time=["t1", "t2", "t3", "t4"],
            values=[1, 2, 3, 4],
            max_lag=3,
        )
