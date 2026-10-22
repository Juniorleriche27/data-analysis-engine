import pytest

from data_analysis_engine import DataAnalysisEngine


def test_linear_trend_recovers_exact_slope():
    result = DataAnalysisEngine().analyze(
        "linear_trend",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
        ],
        values=[
            10,
            12,
            14,
            16,
            18,
        ],
    )

    assert result.result[
        "slope"
    ] == pytest.approx(2.0)

    assert result.result[
        "intercept"
    ] == pytest.approx(10.0)

    assert result.result[
        "r_squared"
    ] == pytest.approx(1.0)


def test_linear_trend_returns_fitted_values():
    result = DataAnalysisEngine().analyze(
        "linear_trend",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
        ],
        values=[
            5,
            8,
            11,
            14,
        ],
    )

    assert result.result[
        "fitted_values"
    ] == pytest.approx(
        [
            5,
            8,
            11,
            14,
        ]
    )


def test_linear_trend_handles_constant_series():
    result = DataAnalysisEngine().analyze(
        "linear_trend",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
        ],
        values=[
            7,
            7,
            7,
            7,
        ],
    )

    assert result.result[
        "slope"
    ] == pytest.approx(0.0)

    assert result.result[
        "r_squared"
    ] is None

    assert (
        result.result[
            "r_squared_defined"
        ]
        is False
    )


def test_linear_detrend_removes_exact_linear_trend():
    result = DataAnalysisEngine().analyze(
        "linear_detrend",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
        ],
        values=[
            10,
            12,
            14,
            16,
            18,
        ],
    )

    assert result.result[
        "detrended"
    ] == pytest.approx(
        [
            0,
            0,
            0,
            0,
            0,
        ]
    )


def test_linear_detrend_residuals_have_zero_mean():
    result = DataAnalysisEngine().analyze(
        "linear_detrend",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
        ],
        values=[
            4,
            9,
            7,
            15,
            12,
        ],
    )

    assert result.result[
        "detrended_mean"
    ] == pytest.approx(
        0.0,
        abs=1e-12,
    )
