from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TaskExecutionResult:
    method: str
    category: str
    columns: list[str]
    status: str

    rows_used: int = 0
    rows_dropped: int = 0

    result: dict[str, Any] | None = None
    error: str | None = None

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "category": self.category,
            "columns": self.columns,
            "status": self.status,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "result": self.result,
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class ExecutionReport:
    dataset_fingerprint: str
    results: list[TaskExecutionResult]

    @property
    def succeeded(self) -> int:
        return sum(
            item.status == "success"
            for item in self.results
        )

    @property
    def failed(self) -> int:
        return sum(
            item.status == "failed"
            for item in self.results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "task_count": len(self.results),
            "succeeded": self.succeeded,
            "failed": self.failed,
            "results": [
                item.to_dict()
                for item in self.results
            ],
        }
