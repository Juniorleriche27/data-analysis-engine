from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    MethodAlreadyRegisteredError,
    MethodNotFoundError,
)


class MethodRegistry:
    """
    Central registry of analytical capabilities.
    """

    def __init__(self) -> None:
        self._methods: dict[str, AnalysisMethod] = {}

    def register(self, method: AnalysisMethod) -> None:
        if method.name in self._methods:
            raise MethodAlreadyRegisteredError(
                f"Method '{method.name}' is already registered."
            )

        self._methods[method.name] = method

    def get(self, name: str) -> AnalysisMethod:
        try:
            return self._methods[name]
        except KeyError as exc:
            raise MethodNotFoundError(
                f"Unknown analytical method: '{name}'."
            ) from exc

    def exists(self, name: str) -> bool:
        return name in self._methods

    def list_methods(self) -> list[dict]:
        return [
            method.metadata()
            for method in self._methods.values()
        ]


registry = MethodRegistry()
