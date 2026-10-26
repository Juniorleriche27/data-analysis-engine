import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_bootstrap_is_reproducible():
    engine = DataAnalysisEngine()

    parameters = {
        "values": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        "n_resamples": 1000,
        "random_state": 7,
    }

    first = engine.analyze(
        "bootstrap_mean_confidence_interval",
        **parameters,
    ).result

    second = engine.analyze(
        "bootstrap_mean_confidence_interval",
        **parameters,
    ).result

    assert first[
        "lower_bound"
    ] == second[
        "lower_bound"
    ]

    assert first[
        "upper_bound"
    ] == second[
        "upper_bound"
    ]


def test_bootstrap_interval_contains_symmetric_sample_mean():
    result = DataAnalysisEngine().analyze(
        "bootstrap_mean_confidence_interval",
        values=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
        n_resamples=2000,
        random_state=42,
    ).result

    assert result[
        "observed_mean"
    ] == pytest.approx(
        3.0
    )

    assert (
        result[
            "lower_bound"
        ]
        <= 3.0
        <= result[
            "upper_bound"
        ]
    )


def test_bootstrap_constant_series_has_zero_width():
    result = DataAnalysisEngine().analyze(
        "bootstrap_mean_confidence_interval",
        values=[
            10.0,
            10.0,
            10.0,
            10.0,
        ],
        n_resamples=500,
    ).result

    assert result[
        "is_constant"
    ] is True

    assert result[
        "lower_bound"
    ] == pytest.approx(
        10.0
    )

    assert result[
        "upper_bound"
    ] == pytest.approx(
        10.0
    )

    assert result[
        "interval_width"
    ] == pytest.approx(
        0.0
    )


def test_bootstrap_preserves_missing_counts():
    result = DataAnalysisEngine().analyze(
        "bootstrap_mean_confidence_interval",
        values=[
            1.0,
            None,
            2.0,
            np.nan,
            3.0,
        ],
        n_resamples=500,
    ).result

    assert result[
        "total_count"
    ] == 5

    assert result[
        "sample_size"
    ] == 3

    assert result[
        "missing_count"
    ] == 2


def test_bootstrap_rejects_too_few_resamples():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "bootstrap_mean_confidence_interval",
            values=[
                1.0,
                2.0,
                3.0,
            ],
            n_resamples=10,
        )
