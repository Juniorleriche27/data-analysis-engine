from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnalysisResult:
    method: str
    category: str
    version: str
    result: dict[str, Any]

    status: str = "success"
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "category": self.category,
            "version": self.version,
            "result": self.result,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
