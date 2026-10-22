from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ColumnProfile:
    name: str
    physical_type: str
    semantic_type: str
    role: str

    count: int
    missing_count: int
    missing_percentage: float
    unique_count: int

    is_constant: bool
    is_unique: bool

    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_type": self.physical_type,
            "semantic_type": self.semantic_type,
            "role": self.role,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "unique_count": self.unique_count,
            "is_constant": self.is_constant,
            "is_unique": self.is_unique,
            "statistics": self.statistics,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_rows: int
    fingerprint: str

    columns: list[ColumnProfile]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "fingerprint": self.fingerprint,
            "columns": [
                column.to_dict()
                for column in self.columns
            ],
            "warnings": self.warnings,
        }
