import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _selection_series():
    return [
        100,
        103,
        106,
        110,
        113,
        117,
        120,
        124,
        128,
        131,
        135,
        139,
        142,
        146,
        150,
    ]


def test_ar_order_selection_returns_lowest_score():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_order_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        min_p=1,
        max_p=3,
        initial_train_size=8,
        metric="mae",
    )

    selection = result.result

    scores = [
        candidate["score"]
        for candidate in selection[
            "candidates"
        ]
        if candidate[
            "score_defined"
        ]
    ]

    assert selection[
        "best_score"
    ] == pytest.approx(
        min(scores)
    )

    best_candidate = next(
        candidate
        for candidate in selection[
            "candidates"
        ]
        if candidate["rank"] == 1
    )

    assert selection[
        "best_order"
    ] == best_candidate["p"]


def test_ar_order_selection_uses_comparable_origins():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_order_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        min_p=1,
        max_p=3,
        initial_train_size=8,
    )

    candidates = result.result[
        "candidates"
    ]

    origin_counts = {
        candidate[
            "origin_count"
        ]
        for candidate in candidates
    }

    forecast_counts = {
        candidate[
            "forecast_observation_count"
        ]
        for candidate in candidates
    }

    assert len(origin_counts) == 1
    assert len(forecast_counts) == 1

    assert (
        result.result[
            "comparable_origins"
        ]
        is True
    )


def test_ar_order_selection_supports_rmse():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_order_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        min_p=1,
        max_p=2,
        initial_train_size=7,
        metric="rmse",
    )

    assert result.result[
        "metric"
    ] == "rmse"

    best = next(
        candidate
        for candidate in result.result[
            "candidates"
        ]
        if candidate["rank"] == 1
    )

    assert result.result[
        "best_score"
    ] == pytest.approx(
        best["metrics"]["rmse"]
    )


def test_ar_order_selection_rejects_invalid_range():
    values = _selection_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "autoregressive_order_selection",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            min_p=3,
            max_p=1,
        )


def test_ar_order_selection_rejects_small_training_window():
    values = _selection_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "autoregressive_order_selection",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            min_p=1,
            max_p=3,
            initial_train_size=6,
        )


def test_ar_order_selection_has_no_lookahead_leakage():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "autoregressive_order_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        min_p=1,
        max_p=2,
        initial_train_size=7,
    )

    assert (
        result.result[
            "look_ahead_leakage"
        ]
        is False
    )

    assert (
        result.result[
            "selection_objective"
        ]
        == "minimize_out_of_sample_metric"
    )
