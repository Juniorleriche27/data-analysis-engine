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


def test_identifier_is_detected():
    dataframe = pd.DataFrame(
        {
            "customer_id": [101, 102, 103, 104],
            "sales": [10, 20, 30, 40],
        }
    )

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)

    columns = {
        column.name: column
        for column in profile.columns
    }

    assert columns["customer_id"].semantic_type == "numeric"
    assert columns["customer_id"].role == "identifier"
    assert columns["customer_id"].statistics == {}

    assert columns["sales"].role == "measure"


def test_datetime_role_is_detected():
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            )
        }
    )

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)

    column = profile.columns[0]

    assert column.semantic_type == "datetime"
    assert column.role == "time"


def test_high_missing_values_warning():
    dataframe = pd.DataFrame(
        {
            "sales": [
                100,
                None,
                None,
                None,
            ]
        }
    )

    dataset = DatasetLoader.from_dataframe(dataframe)
    profile = DataProfiler().profile(dataset)

    warnings = profile.columns[0].warnings

    assert "High percentage of missing values." in warnings
