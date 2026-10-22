from abc import ABC, abstractmethod
from typing import Any


class AnalysisMethod(ABC):
    """
    Base contract implemented by every analytical capability.

    Each method declares both:
    - how it executes;
    - what kind of analytical data it can accept.
    """

    name: str
    category: str
    version: str = "1.0"

    description: str = ""

    supported_roles: tuple[str, ...] = ()
    minimum_compatible_columns: int = 1

    # per_column:
    #   method runs independently on each selected column.
    #
    # pairwise:
    #   method requires pairs of compatible columns.
    #
    # joint:
    #   method consumes several columns simultaneously.
    selection_mode: str = "per_column"

    allow_constant_columns: bool = False

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "description": self.description,
            "supported_roles": list(self.supported_roles),
            "minimum_compatible_columns": self.minimum_compatible_columns,
            "selection_mode": self.selection_mode,
            "allow_constant_columns": self.allow_constant_columns,
        }
