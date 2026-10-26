import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_advanced_quantiles_known_values():
    result = DataAnalysisEngine().analyze(
        "advanced_quantiles",
        values=[
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ],
    ).result

    assert result["q1"] == pytest.approx(2.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["q3"] == pytest.approx(4.0)
    assert result["iqr"] == pytest.approx(2.0)


def test_advanced_quantiles_support_custom_probabilities():
    result = DataAnalysisEngine().analyze(
        "advanced_quantiles",
        values=[
            0.0,
            10.0,
            20.0,
            30.0,
            40.0,
        ],
        probabilities=[
            0.0,
            0.5,
            1.0,
        ],
    ).result

    assert result["quantiles"]["0.0"] == pytest.approx(0.0)
    assert result["quantiles"]["0.5"] == pytest.approx(20.0)
    assert result["quantiles"]["1.0"] == pytest.approx(40.0)


def test_advanced_quantiles_preserve_missing_information():
    result = DataAnalysisEngine().analyze(
        "advanced_quantiles",
        values=[
            1.0,
            None,
            2.0,
            np.nan,
            3.0,
        ],
    ).result

    assert result["total_count"] == 5
    assert result["sample_size"] == 3
    assert result["missing_count"] == 2
    assert result["median"] == pytest.approx(2.0)


def test_advanced_quantiles_reject_invalid_probability():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "advanced_quantiles",
            values=[
                1.0,
                2.0,
                3.0,
            ],
            probabilities=[
                0.5,
                1.5,
            ],
        )
