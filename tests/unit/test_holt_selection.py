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
        105,
        108,
        112,
        114,
        118,
        121,
        125,
        128,
        132,
        135,
        139,
        142,
    ]


def test_holt_parameter_selection_returns_lowest_score():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "holt_parameter_selection",
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
        betas=[
            0.2,
            0.5,
            1.0,
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

    assert selection[
        "best_beta"
    ] == pytest.approx(
        best["beta"]
    )


def test_holt_parameter_selection_finds_expected_pair():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "holt_parameter_selection",
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
        betas=[
            0.2,
            0.5,
            1.0,
        ],
        initial_train_size=7,
        metric="mae",
    )

    assert result.result[
        "best_alpha"
    ] == pytest.approx(
        0.5
    )

    assert result.result[
        "best_beta"
    ] == pytest.approx(
        1.0
    )


def test_holt_parameter_selection_uses_comparable_origins():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "holt_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.5,
            0.8,
        ],
        betas=[
            0.5,
            1.0,
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


def test_holt_parameter_selection_supports_rmse():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "holt_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.5,
            0.8,
        ],
        betas=[
            0.5,
            1.0,
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


def test_holt_parameter_selection_rejects_duplicates():
    values = _selection_series()

    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "holt_parameter_selection",
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
            betas=[
                0.2,
                0.5,
            ],
            initial_train_size=7,
        )


def test_holt_parameter_selection_has_no_lookahead_leakage():
    values = _selection_series()

    result = DataAnalysisEngine().analyze(
        "holt_parameter_selection",
        time=pd.date_range(
            "2026-01-01",
            periods=len(values),
            freq="D",
        ).tolist(),
        values=values,
        alphas=[
            0.5,
            0.8,
        ],
        betas=[
            0.5,
            1.0,
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
