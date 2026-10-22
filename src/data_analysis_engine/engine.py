from typing import Any

from data_analysis_engine.core.bootstrap import register_builtin_methods
from data_analysis_engine.core.registry import registry
from data_analysis_engine.core.result import AnalysisResult


class DataAnalysisEngine:
    """
    Framework-independent entry point.
    """

    def __init__(self) -> None:
        register_builtin_methods()

    def analyze(
        self,
        method_name: str,
        **parameters: Any,
    ) -> AnalysisResult:
        method = registry.get(method_name)

        raw_result = method.execute(**parameters)
        metadata = method.metadata()

        return AnalysisResult(
            method=metadata["name"],
            category=metadata["category"],
            version=metadata["version"],
            result=raw_result,
        )

    def available_methods(self) -> list[dict]:
        return registry.list_methods()
