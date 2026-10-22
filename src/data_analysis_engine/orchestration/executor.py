from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine import DataAnalysisEngine
from data_analysis_engine.core.exceptions import (
    DataAnalysisError,
    InvalidAnalysisInputError,
    PlanDatasetMismatchError,
)
from data_analysis_engine.datasets import Dataset
from data_analysis_engine.orchestration.execution_models import (
    ExecutionReport,
    TaskExecutionResult,
)
from data_analysis_engine.orchestration.planning_models import (
    AnalysisPlan,
    AnalysisTask,
)


class AnalysisExecutor:
    """
    Executes an AnalysisPlan against the Dataset for which
    the plan was created.

    Data binding depends only on selection_mode:
    - per_column -> values
    - pairwise   -> x, y
    - joint      -> data

    Analytical methods remain independent of the executor.
    """

    def __init__(
        self,
        engine: DataAnalysisEngine | None = None,
    ) -> None:
        self.engine = engine or DataAnalysisEngine()

    def execute(
        self,
        dataset: Dataset,
        plan: AnalysisPlan,
        *,
        parameters_by_method: dict[
            str,
            dict[str, Any],
        ]
        | None = None,
    ) -> ExecutionReport:
        current_fingerprint = dataset.fingerprint()

        if current_fingerprint != plan.dataset_fingerprint:
            raise PlanDatasetMismatchError(
                "The analysis plan was created for a different "
                "dataset or dataset version."
            )

        parameters_by_method = (
            parameters_by_method or {}
        )

        results: list[TaskExecutionResult] = []

        for task in plan.tasks:
            try:
                kwargs, rows_used, rows_dropped = (
                    self._bind_task(
                        dataset=dataset,
                        task=task,
                    )
                )

                additional_parameters = (
                    parameters_by_method.get(
                        task.method,
                        {},
                    )
                )

                self._validate_additional_parameters(
                    task=task,
                    parameters=additional_parameters,
                )

                kwargs.update(additional_parameters)

                analysis_result = self.engine.analyze(
                    task.method,
                    **kwargs,
                )

                results.append(
                    TaskExecutionResult(
                        method=task.method,
                        category=task.category,
                        columns=list(task.columns),
                        status="success",
                        rows_used=rows_used,
                        rows_dropped=rows_dropped,
                        result=analysis_result.to_dict(),
                    )
                )

            except DataAnalysisError as exc:
                results.append(
                    TaskExecutionResult(
                        method=task.method,
                        category=task.category,
                        columns=list(task.columns),
                        status="failed",
                        error=str(exc),
                    )
                )

        return ExecutionReport(
            dataset_fingerprint=current_fingerprint,
            results=results,
        )

    def _bind_task(
        self,
        *,
        dataset: Dataset,
        task: AnalysisTask,
    ) -> tuple[dict[str, Any], int, int]:
        dataframe = dataset.data

        missing_columns = [
            column
            for column in task.columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise InvalidAnalysisInputError(
                "Task references missing dataset column(s): "
                + ", ".join(missing_columns)
            )

        selected = dataframe[task.columns]

        original_rows = len(selected)

        if task.selection_mode == "per_column":
            cleaned = selected.dropna()

            values = self._series_to_python_list(
                cleaned.iloc[:, 0]
            )

            return (
                {
                    "values": values,
                },
                len(cleaned),
                original_rows - len(cleaned),
            )

        if task.selection_mode == "pairwise":
            cleaned = selected.dropna()

            x = self._series_to_python_list(
                cleaned.iloc[:, 0]
            )

            y = self._series_to_python_list(
                cleaned.iloc[:, 1]
            )

            return (
                {
                    "x": x,
                    "y": y,
                },
                len(cleaned),
                original_rows - len(cleaned),
            )

        if task.selection_mode == "joint":
            cleaned = selected.dropna()

            data = {
                column: self._series_to_python_list(
                    cleaned[column]
                )
                for column in task.columns
            }

            return (
                {
                    "data": data,
                },
                len(cleaned),
                original_rows - len(cleaned),
            )

        raise InvalidAnalysisInputError(
            "Unsupported task selection mode: "
            f"'{task.selection_mode}'."
        )

    def _validate_additional_parameters(
        self,
        *,
        task: AnalysisTask,
        parameters: dict[str, Any],
    ) -> None:
        if not isinstance(parameters, dict):
            raise InvalidAnalysisInputError(
                "Additional method parameters must be a dictionary."
            )

        reserved_by_mode = {
            "per_column": {"values"},
            "pairwise": {"x", "y"},
            "joint": {"data"},
        }

        reserved = reserved_by_mode.get(
            task.selection_mode,
            set(),
        )

        forbidden = (
            reserved
            & set(parameters.keys())
        )

        if forbidden:
            raise InvalidAnalysisInputError(
                "Additional parameters cannot override "
                "dataset-bound input(s): "
                + ", ".join(sorted(forbidden))
            )

    @staticmethod
    def _series_to_python_list(
        series: pd.Series,
    ) -> list[Any]:
        values: list[Any] = []

        for value in series.tolist():
            if isinstance(value, np.generic):
                value = value.item()

            values.append(value)

        return values
