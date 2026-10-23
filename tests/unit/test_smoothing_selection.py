import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _selection_series():
    return [
        100,
        104,
        102,
        106,
        105,
        109,
        108,
        112,
        111,
        115,
        114,
        118,
        117,
        121,
    ]


def test_ses_alpha_selection_returns_lowest_score():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_alpha_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.2,
            0.5,
            0.8,
        ],
        initial_train_size=7,
        metric="mae",
    )

    selection = result.result

    scores = [
        candidate["score"]
        for candidate in selection[
            "candidates"
        ]
    ]

    assert selection[
        "best_score"
    ] == pytest.approx(
        min(scores)
    )

    best = next(
        candidate
        for candidate in selection[
            "candidates"
        ]
        if candidate["rank"] == 1
    )

    assert selection[
        "best_alpha"
    ] == pytest.approx(
        best["alpha"]
    )


def test_ses_alpha_selection_chooses_expected_candidate():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_alpha_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.2,
            0.5,
            0.8,
        ],
        initial_train_size=7,
        metric="mae",
    )

    assert result.result[
        "best_alpha"
    ] == pytest.approx(
        0.8
    )


def test_ses_alpha_selection_uses_comparable_origins():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_alpha_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.2,
            0.5,
            0.8,
        ],
        initial_train_size=7,
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

    assert len(
        origin_counts
    ) == 1

    assert (
        result.result[
            "comparable_origins"
        ]
        is True
    )


def test_ses_alpha_selection_supports_rmse():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_alpha_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.2,
            0.5,
            0.8,
        ],
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
        best[
            "metrics"
        ]["rmse"]
    )


def test_ses_alpha_selection_rejects_duplicate_candidates():
    values = _selection_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "simple_exponential_smoothing_alpha_selection",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            alphas=[
                0.5,
                0.5,
            ],
            initial_train_size=7,
        )


def test_ses_alpha_selection_has_no_lookahead_leakage():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "simple_exponential_smoothing_alpha_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.2,
            0.5,
            0.8,
        ],
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
