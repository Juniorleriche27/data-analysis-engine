import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_temporal_quality_detects_clean_regular_axis():
    values = pd.date_range(
        "2026-01-01",
        periods=5,
        freq="h",
    )

    result = DataAnalysisEngine().analyze(
        "temporal_quality_diagnostic",
        values=values,
    ).result

    assert result[
        "parseable_count"
    ] == 5

    assert result[
        "duplicate_timestamp_count"
    ] == 0

    assert (
        result[
            "is_strictly_increasing"
        ]
        is True
    )

    assert (
        result[
            "is_regular"
        ]
        is True
    )

    assert result[
        "median_interval_seconds"
    ] == pytest.approx(
        3600.0
    )

    assert result[
        "risk_level"
    ] == "none"


def test_temporal_quality_detects_unparseable_values():
    result = DataAnalysisEngine().analyze(
        "temporal_quality_diagnostic",
        values=[
            "2026-01-01",
            "not-a-date",
            "2026-01-03",
        ],
    ).result

    assert result[
        "unparseable_count"
    ] == 1

    assert result[
        "parse_success_rate"
    ] == pytest.approx(
        2 / 3
    )

    assert result[
        "quality_status"
    ] == "unparseable_values"

    assert result[
        "risk_level"
    ] == "high"


def test_temporal_quality_detects_duplicate_timestamps():
    result = DataAnalysisEngine().analyze(
        "temporal_quality_diagnostic",
        values=[
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-03",
        ],
    ).result

    assert result[
        "duplicate_timestamp_count"
    ] == 1

    assert (
        result[
            "is_strictly_increasing"
        ]
        is False
    )

    assert result[
        "quality_status"
    ] == "duplicate_timestamps"


def test_temporal_quality_detects_out_of_order_axis():
    result = DataAnalysisEngine().analyze(
        "temporal_quality_diagnostic",
        values=[
            "2026-01-01",
            "2026-01-03",
            "2026-01-02",
            "2026-01-04",
        ],
    ).result

    assert result[
        "out_of_order_count"
    ] == 1

    assert (
        result[
            "is_monotonic_increasing"
        ]
        is False
    )

    assert result[
        "quality_status"
    ] == "out_of_order"


def test_temporal_quality_detects_irregular_intervals():
    result = DataAnalysisEngine().analyze(
        "temporal_quality_diagnostic",
        values=[
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-10",
        ],
        tolerance_ratio=0.05,
    ).result

    assert result[
        "irregular_interval_count"
    ] == 1

    assert result[
        "gap_count"
    ] == 1

    assert (
        result[
            "is_regular"
        ]
        is False
    )

    assert result[
        "quality_status"
    ] == "irregular_intervals"


def test_temporal_quality_preserves_missing_information():
    result = DataAnalysisEngine().analyze(
        "temporal_quality_diagnostic",
        values=[
            "2026-01-01",
            None,
            "2026-01-02",
            pd.NA,
        ],
    ).result

    assert result[
        "total_count"
    ] == 4

    assert result[
        "non_missing_count"
    ] == 2

    assert result[
        "missing_count"
    ] == 2

    assert result[
        "parseable_count"
    ] == 2


def test_temporal_quality_rejects_invalid_tolerance():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "temporal_quality_diagnostic",
            values=[
                "2026-01-01",
                "2026-01-02",
            ],
            tolerance_ratio=1.0,
        )
