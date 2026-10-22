import pandas as pd
import pytest

from data_analysis_engine.datasets import DatasetLoader
from data_analysis_engine.profiling import DataProfiler


def test_profile_basic_dataset():
    dataframe = pd.DataFrame(
        {
            "sales": [100, 120, 140, 160],
            "region": ["North", "South", "North", "South"],
            "active": [True, True, False, True],
        }
    )

    dataset = DatasetLoader.from_dataframe(
        dataframe,
        name="sales",
    )

    profile = DataProfiler().profile(dataset)

    assert profile.row_count == 4
    assert profile.column_count == 3
    assert profile.duplicate_rows == 0

    columns = {
        column.name: column
        for column in profile.columns
    }

    assert columns["sales"].semantic_type == "numeric"
    assert columns["sales"].statistics["mean"] == pytest.approx(130.0)

    assert columns["region"].semantic_type == "categorical"
    assert columns["active"].semantic_type == "boolean"


def test_missing_values_are_detected():
    dataframe = pd.DataFrame(
        {
            "sales": [100, None, 300],
        }
    )

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)

    column = profile.columns[0]

    assert column.missing_count == 1
    assert column.missing_percentage == pytest.approx(
        33.3333333333
    )


def test_duplicate_rows_are_detected():
    dataframe = pd.DataFrame(
        {
            "x": [1, 1, 2],
            "y": ["A", "A", "B"],
        }
    )

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)

    assert profile.duplicate_rows == 1


def test_dataset_fingerprint_is_stable():
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3],
        }
    )

    first = DatasetLoader.from_dataframe(dataframe)
    second = DatasetLoader.from_dataframe(dataframe)

    assert first.fingerprint() == second.fingerprint()


def test_unique_small_text_column_is_text():
    dataframe = pd.DataFrame(
        {
            "comment": [
                "Excellent service",
                "Delivery was late",
                "I want a refund",
                "Very satisfied",
            ]
        }
    )

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)

    assert profile.columns[0].semantic_type == "text"
