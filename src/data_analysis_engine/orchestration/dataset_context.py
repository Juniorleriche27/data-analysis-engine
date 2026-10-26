from __future__ import annotations

from typing import Any

import pandas as pd


def build_safe_dataset_context(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "'dataframe' must be a pandas DataFrame."
        )

    columns = []

    for column in dataframe.columns:
        series = dataframe[column]

        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "row_count": int(
                    len(series)
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "non_missing_count": int(
                    series.notna().sum()
                ),
                "unique_non_missing_count": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return {
        "row_count": int(
            len(dataframe)
        ),
        "column_count": int(
            len(dataframe.columns)
        ),
        "columns": columns,
        "contains_raw_values": False,
    }
