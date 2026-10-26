from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

from data_analysis_engine.orchestration.ai_contracts import (
    AIAnalysisPlan,
)

if TYPE_CHECKING:
    from data_analysis_engine import (
        DataAnalysisEngine,
    )


class DeterministicAIPlanExecutor:
    """
    AI chooses methods and parameters.

    All analytical calculations are delegated to
    DataAnalysisEngine.
    """

    def __init__(
        self,
        engine: "DataAnalysisEngine | None" = None,
    ) -> None:
        if engine is None:
            from data_analysis_engine import (
                DataAnalysisEngine,
            )

            engine = DataAnalysisEngine()

        self.engine = engine

    def registered_methods(
        self,
    ) -> set[str]:
        return {
            method[
                "name"
            ]
            for method
            in self.engine.available_methods()
        }

    def validate_plan(
        self,
        plan: (
            AIAnalysisPlan
            | Mapping[str, Any]
        ),
    ) -> AIAnalysisPlan:
        if isinstance(
            plan,
            AIAnalysisPlan,
        ):
            registered = (
                self.registered_methods()
            )

            for call in plan.calls:
                if (
                    call.method
                    not in registered
                ):
                    raise ValueError(
                        "Plan contains an "
                        "unregistered method."
                    )

            return plan

        return AIAnalysisPlan.from_mapping(
            plan,
            registered_methods=(
                self.registered_methods()
            ),
        )

    def execute(
        self,
        plan: (
            AIAnalysisPlan
            | Mapping[str, Any]
        ),
    ) -> dict[str, Any]:
        validated = self.validate_plan(
            plan
        )

        calls = []

        for call in validated.calls:
            response = self.engine.analyze(
                call.method,
                **call.parameters,
            )

            calls.append(
                {
                    "call_id": (
                        call.call_id
                    ),
                    "method": (
                        call.method
                    ),
                    "status": "success",
                    "result": (
                        response.result
                    ),
                }
            )

        return {
            "plan_id": (
                validated.plan_id
            ),
            "execution_policy": (
                "deterministic_engine_only"
            ),
            "call_count": len(
                calls
            ),
            "calls": calls,
        }
