import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_rolling_zscore_detects_spike():
    result = DataAnalysisEngine().analyze(
        "rolling_zscore_anomaly",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
        ],
        values=[
            10,
            10,
            10,
            10,
            50,
        ],
        window=3,
    )

    records = result.result[
        "records"
    ]

    assert records[-1][
        "anomaly"
    ] is True

    assert records[-1][
        "timestamp"
    ] == "t5"


def test_rolling_zscore_uses_only_previous_observations():
    result = DataAnalysisEngine().analyze(
        "rolling_zscore_anomaly",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
        ],
        values=[
            10,
            10,
            10,
            10,
            100,
        ],
        window=3,
    )

    spike = result.result[
        "records"
    ][-1]

    assert spike[
        "baseline_mean"
    ] == pytest.approx(10.0)

    assert spike[
        "baseline_end_index"
    ] == 3

    assert (
        result.result[
            "baseline_policy"
        ]
        == "previous_observations_only"
    )


def test_rolling_zscore_computes_numeric_score():
    result = DataAnalysisEngine().analyze(
        "rolling_zscore_anomaly",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
        ],
        values=[
            10,
            12,
            11,
            13,
        ],
        window=3,
        threshold=1.5,
    )

    record = result.result[
        "records"
    ][0]

    assert record[
        "z_score"
    ] == pytest.approx(2.0)

    assert record[
        "anomaly"
    ] is True


def test_rolling_mad_detects_robust_spike():
    result = DataAnalysisEngine().analyze(
        "rolling_mad_anomaly",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
            "t5",
        ],
        values=[
            9,
            10,
            11,
            10,
            50,
        ],
        window=4,
    )

    record = result.result[
        "records"
    ][0]

    assert record[
        "baseline_median"
    ] == pytest.approx(10.0)

    assert record[
        "baseline_mad"
    ] == pytest.approx(0.5)

    assert record[
        "anomaly"
    ] is True


def test_rolling_mad_handles_zero_mad_baseline():
    result = DataAnalysisEngine().analyze(
        "rolling_mad_anomaly",
        time=[
            "t1",
            "t2",
            "t3",
            "t4",
        ],
        values=[
            5,
            5,
            5,
            20,
        ],
        window=3,
    )

    record = result.result[
        "records"
    ][0]

    assert record[
        "score_defined"
    ] is False

    assert record[
        "anomaly"
    ] is True

    assert (
        record["reason"]
        == "departure_from_zero_mad_baseline"
    )


def test_anomaly_detection_rejects_invalid_window():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "rolling_zscore_anomaly",
            time=[
                "t1",
                "t2",
                "t3",
            ],
            values=[
                1,
                2,
                3,
            ],
            window=3,
        )
