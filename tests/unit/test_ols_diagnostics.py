import numpy as np
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


# ============================================================
# BREUSCH-PAGAN
# ============================================================

def test_breusch_pagan_returns_valid_statistics():
    result = DataAnalysisEngine().analyze(
        "breusch_pagan_heteroskedasticity_test",
        data={
            "x": list(
                range(
                    1,
                    21,
                )
            ),
            "y": [
                2.2,
                4.1,
                6.3,
                7.8,
                10.2,
                12.1,
                13.9,
                16.3,
                17.8,
                20.4,
                22.1,
                23.9,
                26.2,
                27.8,
                30.1,
                32.4,
                33.9,
                36.2,
                37.8,
                40.1,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "lm_statistic"
    ] >= 0.0

    assert (
        0.0
        <= result[
            "p_value"
        ]
        <= 1.0
    )


def test_breusch_pagan_detects_strong_heteroskedasticity():
    x = np.arange(
        1,
        41,
        dtype=float,
    )

    errors = (
        (
            -1.0
        )
        ** np.arange(
            40
        )
        * 0.8
        * x
    )

    y = (
        2.0
        * x
        + errors
    )

    result = DataAnalysisEngine().analyze(
        "breusch_pagan_heteroskedasticity_test",
        data={
            "x": x,
            "y": y,
        },
        target_column="y",
    ).result

    assert result[
        "heteroskedasticity_detected"
    ] is True

    assert result[
        "p_value"
    ] < 0.05


def test_breusch_pagan_tracks_dropped_rows():
    result = DataAnalysisEngine().analyze(
        "breusch_pagan_heteroskedasticity_test",
        data={
            "x": [
                1.0,
                2.0,
                None,
                4.0,
                5.0,
                6.0,
            ],
            "y": [
                2.1,
                4.2,
                6.0,
                None,
                10.3,
                12.1,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "sample_size"
    ] == 4

    assert result[
        "dropped_count"
    ] == 2


def test_breusch_pagan_rejects_invalid_significance_level():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "breusch_pagan_heteroskedasticity_test",
            data={
                "x": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
                "y": [
                    2.0,
                    4.0,
                    6.1,
                    8.0,
                ],
            },
            target_column="y",
            significance_level=1.0,
        )


# ============================================================
# DURBIN-WATSON
# ============================================================

def test_residual_autocorrelation_returns_durbin_watson():
    result = DataAnalysisEngine().analyze(
        "ols_residual_autocorrelation_diagnostic",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            "y": [
                2.2,
                4.1,
                6.3,
                8.2,
                9.8,
                11.9,
                14.2,
                15.8,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "durbin_watson"
    ] is not None

    assert (
        0.0
        <= result[
            "durbin_watson"
        ]
        <= 4.0
    )


def test_residual_autocorrelation_reports_perfect_fit():
    result = DataAnalysisEngine().analyze(
        "ols_residual_autocorrelation_diagnostic",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "y": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "perfect_fit"
    ] is True

    assert result[
        "durbin_watson"
    ] is None

    assert result[
        "interpretation"
    ] == "perfect_fit"


def test_residual_autocorrelation_is_order_sensitive():
    result = DataAnalysisEngine().analyze(
        "ols_residual_autocorrelation_diagnostic",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "y": [
                2.0,
                4.3,
                5.7,
                8.5,
                9.6,
                12.2,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "order_sensitive"
    ] is True


def test_residual_autocorrelation_tracks_missing_rows():
    result = DataAnalysisEngine().analyze(
        "ols_residual_autocorrelation_diagnostic",
        data={
            "x": [
                1.0,
                None,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "y": [
                2.2,
                4.0,
                6.1,
                None,
                10.4,
                12.1,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "sample_size"
    ] == 4

    assert result[
        "dropped_count"
    ] == 2


# ============================================================
# INFLUENCE
# ============================================================

def test_influence_returns_leverage_and_cooks_distance():
    result = DataAnalysisEngine().analyze(
        "ols_influence_diagnostic",
        data={
            "x": list(
                range(
                    1,
                    11,
                )
            ),
            "y": [
                2.1,
                3.9,
                6.2,
                8.1,
                9.8,
                12.2,
                13.9,
                16.1,
                17.8,
                20.2,
            ],
        },
        target_column="y",
    ).result

    assert len(
        result[
            "leverage"
        ]
    ) == 10

    assert len(
        result[
            "cooks_distance"
        ]
    ) == 10

    assert sum(
        result[
            "leverage"
        ]
    ) == pytest.approx(
        result[
            "parameter_count"
        ]
    )


def test_influence_detects_large_outlier():
    x = np.arange(
        1,
        11,
        dtype=float,
    )

    y = (
        2.0
        * x
    )

    y[
        -1
    ] = 50.0

    result = DataAnalysisEngine().analyze(
        "ols_influence_diagnostic",
        data={
            "x": x,
            "y": y,
        },
        target_column="y",
    ).result

    assert 9 in result[
        "influential_positions"
    ]

    assert result[
        "influential_count"
    ] >= 1


def test_influence_thresholds_are_reported():
    result = DataAnalysisEngine().analyze(
        "ols_influence_diagnostic",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "y": [
                2.1,
                4.2,
                5.8,
                8.1,
                9.9,
                12.2,
            ],
        },
        target_column="y",
    ).result

    assert result[
        "high_leverage_threshold"
    ] > 0.0

    assert result[
        "cooks_distance_threshold"
    ] == pytest.approx(
        4.0 / 6.0
    )


def test_influence_handles_perfect_fit():
    result = DataAnalysisEngine().analyze(
        "ols_influence_diagnostic",
        data={
            "x": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "y": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
            ],
        },
        target_column="y",
    ).result

    assert all(
        value == pytest.approx(
            0.0
        )
        for value
        in result[
            "cooks_distance"
        ]
    )
