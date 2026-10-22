from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AnalysisTask:
    """
    One concrete analytical operation to execute.
    """

    method: str
    category: str
    selection_mode: str
    columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "category": self.category,
            "selection_mode": self.selection_mode,
            "columns": self.columns,
        }


@dataclass(slots=True)
class AnalysisPlan:
    """
    Collection of analytical tasks generated for one dataset.
    """

    dataset_fingerprint: str
    tasks: list[AnalysisTask]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "task_count": len(self.tasks),
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ],
        }
