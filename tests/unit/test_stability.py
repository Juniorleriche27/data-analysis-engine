import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_stability_diagnostic_is_explicitly_not_formal():
    result = DataAnalysisEngine().analyze(
        "time_series_stability_diagnostic",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
        ],
        values=[
            3,
            4,
            2,
            5,
            3,
            4,
        ],
    )

    assert (
        result.result[
            "formal_stationarity_test"
        ]
        is False
    )

    assert (
        result.result[
            "diagnostic_type"
        ]
        == "descriptive"
    )


def test_constant_series_has_no_instability_signal():
    result = DataAnalysisEngine().analyze(
        "time_series_stability_diagnostic",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
            "t6",
        ],
        values=[
            10,
            10,
            10,
            10,
            10,
            10,
        ],
    )

    assert (
        result.result["assessment"]
        == "no_instability_signal"
    )

    assert (
        result.result[
            "instability_signals"
        ]
        == []
    )


def test_strong_linear_trend_is_detected():
    result = DataAnalysisEngine().analyze(
        "time_series_stability_diagnostic",
        time=[
            f"t{index}"
            for index in range(
                1,
                11,
            )
        ],
        values=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
    )

    assert (
        result.result["assessment"]
        == "instability_signals_detected"
    )

    assert (
        "trend_stability"
        in result.result[
            "instability_signals"
        ]
    )


def test_stability_diagnostic_detects_variance_change():
    result = DataAnalysisEngine().analyze(
        "time_series_stability_diagnostic",
        time=[
            f"t{index}"
            for index in range(
                1,
                13,
            )
        ],
        values=[
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -10,
            10,
            -10,
            10,
            -10,
            10,
        ],
    )

    assert (
        "variance_stability"
        in result.result[
            "instability_signals"
        ]
    )


def test_stability_thresholds_are_configurable():
    result = DataAnalysisEngine().analyze(
        "time_series_stability_diagnostic",
        time=[
            f"t{index}"
            for index in range(
                1,
                11,
            )
        ],
        values=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ],
        max_standardized_trend_strength=100.0,
        max_standardized_mean_shift=100.0,
        max_variance_ratio=100.0,
        max_abs_lag1_autocorrelation=1.1,
    )

    assert (
        result.result["assessment"]
        == "no_instability_signal"
    )


def test_stability_diagnostic_requires_sufficient_data():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "time_series_stability_diagnostic",
            time=[
                "t1",
                "t2",
                "t3",
                "t4",
                "t5",
            ],
            values=[
                1,
                2,
                3,
                4,
                5,
            ],
        )
