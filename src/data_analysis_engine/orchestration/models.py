from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RoutingCandidate:
    method: str
    category: str
    selection_mode: str
    compatible_columns: list[str]

    compatible_column_roles: dict[str, str] = field(
        default_factory=dict
    )

    role_requirements: dict[str, int] = field(
        default_factory=dict
    )

    reasons: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "category": self.category,
            "selection_mode": self.selection_mode,
            "compatible_columns": self.compatible_columns,
            "compatible_column_roles": self.compatible_column_roles,
            "role_requirements": self.role_requirements,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class RejectedMethod:
    method: str
    category: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "category": self.category,
            "reasons": self.reasons,
        }


@dataclass(slots=True)
class RoutingReport:
    dataset_fingerprint: str

    candidates: list[RoutingCandidate]
    rejected: list[RejectedMethod]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "rejected": [
                rejection.to_dict()
                for rejection in self.rejected
            ],
        }
