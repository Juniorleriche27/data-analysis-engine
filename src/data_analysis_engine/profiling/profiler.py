from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.datasets.dataset import Dataset
from data_analysis_engine.profiling.config import ProfilerConfig
from data_analysis_engine.profiling.models import (
    ColumnProfile,
    DatasetProfile,
)


class DataProfiler:
    """
    Inspects datasets before analytical methods are selected.
    """

    def __init__(
        self,
        config: ProfilerConfig | None = None,
    ) -> None:
        self.config = config or ProfilerConfig()

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
            warnings.append(
                "Dataset contains no rows."
            )

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
        non_missing = series.dropna()

        missing_count = int(
            series.isna().sum()
        )

        missing_percentage = (
            (missing_count / row_count) * 100
            if row_count
            else 0.0
        )

        unique_count = int(
            non_missing.nunique()
        )

        is_unique = (
            unique_count == len(non_missing)
            and len(non_missing) > 0
        )

        is_constant = (
            unique_count <= 1
            and len(non_missing) > 0
        )

        semantic_type = self._infer_semantic_type(
            series
        )

        role = self._infer_role(
            name=name,
            semantic_type=semantic_type,
            is_unique=is_unique,
        )

        warnings: list[str] = []

        if missing_count:
            warnings.append(
                f"{missing_count} missing value(s)."
            )

        if (
            missing_percentage
            >= self.config.high_missing_percentage_warning
        ):
            warnings.append(
                "High percentage of missing values."
            )

        if is_constant:
            warnings.append(
                "Column is constant."
            )

        if role == "identifier":
            warnings.append(
                "Column appears to be an identifier."
            )

        statistics = self._statistics(
            series=series,
            semantic_type=semantic_type,
            role=role,
        )

        return ColumnProfile(
            name=name,
            physical_type=str(series.dtype),
            semantic_type=semantic_type,
            role=role,
            count=row_count,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
            is_constant=is_constant,
            is_unique=is_unique,
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

        unique_count = int(
            non_missing.nunique()
        )

        row_count = len(non_missing)

        unique_ratio = (
            unique_count / row_count
            if row_count
            else 0.0
        )

        if (
            row_count
            <= self.config.small_dataset_max_rows
        ):
            if (
                unique_ratio
                <= self.config.small_dataset_categorical_ratio_threshold
            ):
                return "categorical"

        elif (
            unique_ratio
            <= self.config.categorical_ratio_threshold
        ):
            return "categorical"

        return "text"

    def _infer_role(
        self,
        *,
        name: str,
        semantic_type: str,
        is_unique: bool,
    ) -> str:
        normalized_name = name.lower().strip()

        identifier_names = {
            "id",
            "uuid",
            "guid",
            "identifier",
        }

        looks_like_identifier = (
            normalized_name in identifier_names
            or normalized_name.endswith("_id")
            or normalized_name.startswith("id_")
        )

        if is_unique and looks_like_identifier:
            return "identifier"

        if semantic_type == "datetime":
            return "time"

        if semantic_type == "numeric":
            return "measure"

        if semantic_type in {
            "categorical",
            "boolean",
        }:
            return "dimension"

        if semantic_type == "text":
            return "text"

        return "unknown"

    def _statistics(
        self,
        *,
        series: pd.Series,
        semantic_type: str,
        role: str,
    ) -> dict[str, Any]:
        non_missing = series.dropna()

        if non_missing.empty:
            return {}

        if (
            semantic_type == "numeric"
            and role != "identifier"
        ):
            numeric_values = pd.to_numeric(
                non_missing,
                errors="coerce",
            )

            finite_values = numeric_values[
                np.isfinite(
                    numeric_values
                )
            ]

            if finite_values.empty:
                return {}

            return {
                "mean": float(finite_values.mean()),
                "median": float(finite_values.median()),
                "std": (
                    float(finite_values.std())
                    if len(finite_values) > 1
                    else 0.0
                ),
                "minimum": float(finite_values.min()),
                "maximum": float(finite_values.max()),
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
            counts = (
                non_missing
                .astype(str)
                .value_counts()
            )

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
