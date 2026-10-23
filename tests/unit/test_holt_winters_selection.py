import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def _series():
    return [
        110,
        97,
        109,
        96,
        118,
        105,
        117,
        104,
        127,
        112,
        126,
        111,
        133,
        122,
        134,
        119,
        143,
        128,
        142,
        129,
    ]


def test_hw_parameter_selection_returns_lowest_score():
    values = _series()

    result = DataAnalysisEngine().analyze(
        "holt_winters_additive_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        season_length=4,
        alphas=[
            0.5,
            1.0,
        ],
        betas=[
            0.5,
            1.0,
        ],
        gammas=[
            0.5,
            1.0,
        ],
        initial_train_size=8,
        metric="mae",
    ).result

    scores = [
        candidate["score"]
        for candidate in result[
            "candidates"
        ]
    ]

    assert result[
        "best_score"
    ] == pytest.approx(
        min(scores)
    )

    best = next(
        candidate
        for candidate in result[
            "candidates"
        ]
        if candidate["rank"] == 1
    )

    assert result[
        "best_parameters"
    ] == {
        "alpha": best["alpha"],
        "beta": best["beta"],
        "gamma": best["gamma"],
        "season_length": 4,
    }


def test_hw_parameter_selection_counts_grid():
    values = _series()

    result = DataAnalysisEngine().analyze(
        "holt_winters_additive_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        season_length=4,
        alphas=[
            0.5,
            1.0,
        ],
        betas=[
            0.5,
            1.0,
        ],
        gammas=[
            0.2,
            0.5,
            1.0,
        ],
        initial_train_size=8,
    ).result

    assert result[
        "parameter_combinations_evaluated"
    ] == 12

    assert result[
        "alpha_candidates"
    ] == 2

    assert result[
        "beta_candidates"
    ] == 2

    assert result[
        "gamma_candidates"
    ] == 3


def test_hw_parameter_selection_uses_comparable_origins():
    values = _series()

    result = DataAnalysisEngine().analyze(
        "holt_winters_additive_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        season_length=4,
        alphas=[
            0.5,
            1.0,
        ],
        betas=[
            0.5,
            1.0,
        ],
        gammas=[
            0.5,
            1.0,
        ],
        initial_train_size=8,
    ).result

    origin_counts = {
        candidate[
            "origin_count"
        ]
        for candidate in result[
            "candidates"
        ]
    }

    assert len(
        origin_counts
    ) == 1

    assert (
        result[
            "comparable_origins"
        ]
        is True
    )

    assert (
        result[
            "look_ahead_leakage"
        ]
        is False
    )


def test_hw_parameter_selection_supports_rmse():
    values = _series()

    result = DataAnalysisEngine().analyze(
        "holt_winters_additive_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        season_length=4,
        alphas=[
            0.5,
            1.0,
        ],
        betas=[
            0.5,
            1.0,
        ],
        gammas=[
            0.5,
            1.0,
        ],
        initial_train_size=8,
        metric="rmse",
    ).result

    assert result[
        "metric"
    ] == "rmse"

    best = next(
        candidate
        for candidate in result[
            "candidates"
        ]
        if candidate["rank"] == 1
    )

    assert result[
        "best_score"
    ] == pytest.approx(
        best[
            "metrics"
        ]["rmse"]
    )


def test_hw_parameter_selection_rejects_duplicates():
    values = _series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "holt_winters_additive_parameter_selection",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            season_length=4,
            alphas=[
                0.5,
            ],
            betas=[
                0.5,
            ],
            gammas=[
                0.5,
                0.5,
            ],
            initial_train_size=8,
        )


def test_hw_parameter_selection_rejects_small_training_window():
    values = _series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "holt_winters_additive_parameter_selection",
            time=pd.date_range(
                "2026-01-01",
                periods=len(values),
                freq="D",
            ).tolist(),
            values=values,
            season_length=4,
            alphas=[
                0.5,
            ],
            betas=[
                0.5,
            ],
            gammas=[
                0.5,
            ],
            initial_train_size=7,
        )
