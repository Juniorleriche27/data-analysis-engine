import pandas as pd
import pytest

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


def test_regular_daily_time_axis():
    result = DataAnalysisEngine().analyze(
        "temporal_structure_diagnostic",
        time=pd.date_range(
            "2026-01-01",
            periods=5,
            freq="D",
        ).tolist(),
    )

    diagnostic = result.result

    assert diagnostic[
        "status"
    ] == "regular"

    assert diagnostic[
        "expected_interval_seconds"
    ] == pytest.approx(
        86400.0
    )

    assert diagnostic[
        "regularity_ratio"
    ] == pytest.approx(
        1.0
    )

    assert diagnostic[
        "duplicate_observation_count"
    ] == 0

    assert diagnostic[
        "gap_count"
    ] == 0


def test_temporal_structure_detects_unsorted_input():
    result = DataAnalysisEngine().analyze(
        "temporal_structure_diagnostic",
        time=pd.to_datetime(
            [
                "2026-01-03",
                "2026-01-01",
                "2026-01-02",
            ]
        ).tolist(),
    )

    diagnostic = result.result

    assert diagnostic[
        "input_monotonic_increasing"
    ] is False

    assert diagnostic[
        "status"
    ] == "regular"


def test_temporal_structure_detects_duplicate_timestamp():
    result = DataAnalysisEngine().analyze(
        "temporal_structure_diagnostic",
        time=pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-02",
                "2026-01-03",
            ]
        ).tolist(),
    )

    diagnostic = result.result

    assert diagnostic[
        "duplicate_observation_count"
    ] == 1

    assert diagnostic[
        "input_strictly_increasing"
    ] is False

    assert any(
        record["type"] == "duplicate"
        for record in diagnostic[
            "intervals"
        ]
    )


def test_temporal_structure_detects_gap():
    result = DataAnalysisEngine().analyze(
        "temporal_structure_diagnostic",
        time=pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-05",
                "2026-01-06",
            ]
        ).tolist(),
    )

    diagnostic = result.result

    assert diagnostic[
        "expected_interval_seconds"
    ] == pytest.approx(
        86400.0
    )

    assert diagnostic[
        "gap_count"
    ] == 1

    assert diagnostic[
        "estimated_missing_periods"
    ] == 1


def test_temporal_structure_rejects_invalid_timestamps():
    with pytest.raises(
        InvalidAnalysisInputError
    ):
        DataAnalysisEngine().analyze(
            "temporal_structure_diagnostic",
            time=[
                "2026-01-01",
                "not-a-date",
                "2026-01-03",
            ],
        )
