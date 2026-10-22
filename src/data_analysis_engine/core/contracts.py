from abc import ABC, abstractmethod
from typing import Any


class AnalysisMethod(ABC):
    """
    Base contract implemented by every analytical capability.
    """

    name: str
    category: str
    version: str = "1.0"

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "version": self.version,
        }
