from typing import Any

import pandas as pd

from data_analysis_engine.datasets.dataset import Dataset
from data_analysis_engine.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)


class DataProfiler:
    """
    Inspects a dataset before analytical methods are selected.
    """

    def profile(self, dataset: Dataset) -> DatasetProfile:
        dataframe = dataset.data

        columns = [
            self._profile_column(
                name=str(column),
                series=dataframe[column],
            )
            for column in dataframe.columns
        ]

        warnings: list[str] = []

        duplicate_rows = int(
            dataframe.duplicated().sum()
        )

        if duplicate_rows:
            warnings.append(
                f"Dataset contains {duplicate_rows} duplicate row(s)."
            )

        if dataframe.empty:
            warnings.append("Dataset contains no rows.")

        return DatasetProfile(
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            duplicate_rows=duplicate_rows,
            fingerprint=dataset.fingerprint(),
            columns=columns,
            warnings=warnings,
        )

    def _profile_column(
        self,
        *,
        name: str,
        series: pd.Series,
    ) -> ColumnProfile:
        row_count = len(series)

        missing_count = int(series.isna().sum())

        missing_percentage = (
            (missing_count / row_count) * 100
            if row_count
            else 0.0
        )

        non_missing = series.dropna()

        unique_count = int(
            non_missing.nunique()
        )

        semantic_type = self._infer_semantic_type(
            series
        )

        warnings: list[str] = []

        if missing_count:
            warnings.append(
                f"{missing_count} missing value(s)."
            )

        if unique_count <= 1 and len(non_missing) > 0:
            warnings.append(
                "Column is constant."
            )

        statistics = self._statistics(
            series=series,
            semantic_type=semantic_type,
        )

        return ColumnProfile(
            name=name,
            physical_type=str(series.dtype),
            semantic_type=semantic_type,
            count=row_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            is_constant=(
                unique_count <= 1
                and len(non_missing) > 0
            ),
            is_unique=(
                unique_count == len(non_missing)
                and len(non_missing) > 0
            ),
            statistics=statistics,
            warnings=warnings,
        )

    def _infer_semantic_type(
        self,
        series: pd.Series,
    ) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        non_missing = series.dropna()

        if non_missing.empty:
            return "unknown"

        unique_count = int(non_missing.nunique())
        row_count = len(non_missing)

        unique_ratio = (
            unique_count / row_count
            if row_count
            else 0.0
        )

        # Small datasets need a more tolerant threshold.
        # Example:
        # ["North", "South", "North", "South"]
        # has 50% unique values but is clearly categorical.
        if row_count <= 20:
            if unique_ratio <= 0.50:
                return "categorical"
        elif unique_ratio <= 0.20:
            return "categorical"

        return "text"

    def _statistics(
        self,
        *,
        series: pd.Series,
        semantic_type: str,
    ) -> dict[str, Any]:
        non_missing = series.dropna()

        if non_missing.empty:
            return {}

        if semantic_type == "numeric":
            return {
                "mean": float(non_missing.mean()),
                "median": float(non_missing.median()),
                "std": (
                    float(non_missing.std())
                    if len(non_missing) > 1
                    else 0.0
                ),
                "minimum": float(non_missing.min()),
                "maximum": float(non_missing.max()),
            }

        if semantic_type == "datetime":
            return {
                "minimum": non_missing.min().isoformat(),
                "maximum": non_missing.max().isoformat(),
            }

        if semantic_type in {
            "categorical",
            "boolean",
            "text",
        }:
            counts = non_missing.astype(str).value_counts()

            return {
                "most_frequent": (
                    str(counts.index[0])
                    if not counts.empty
                    else None
                ),
                "most_frequent_count": (
                    int(counts.iloc[0])
                    if not counts.empty
                    else 0
                ),
            }

        return {}
