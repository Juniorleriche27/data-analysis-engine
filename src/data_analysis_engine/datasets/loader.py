from pathlib import Path

import pandas as pd

from data_analysis_engine.core.exceptions import InvalidAnalysisInputError
from data_analysis_engine.datasets.dataset import Dataset


class DatasetLoader:
    """
    Entry point for loading external tabular data
    into the engine's Dataset representation.
    """

    @staticmethod
    def from_dataframe(
        dataframe: pd.DataFrame,
        *,
        name: str | None = None,
        source: str | None = None,
    ) -> Dataset:
        if not isinstance(dataframe, pd.DataFrame):
            raise InvalidAnalysisInputError(
                "Expected a pandas DataFrame."
            )

        return Dataset(
            data=dataframe.copy(),
            name=name,
            source=source,
        )

    @staticmethod
    def from_csv(
        path: str | Path,
        *,
        name: str | None = None,
        **read_csv_options,
    ) -> Dataset:
        path = Path(path)

        if not path.exists():
            raise InvalidAnalysisInputError(
                f"CSV file does not exist: {path}"
            )

        if not path.is_file():
            raise InvalidAnalysisInputError(
                f"CSV path is not a file: {path}"
            )

        dataframe = pd.read_csv(
            path,
            **read_csv_options,
        )

        return Dataset(
            data=dataframe,
            name=name or path.stem,
            source=str(path.resolve()),
        )
