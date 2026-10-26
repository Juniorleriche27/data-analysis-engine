from __future__ import annotations

from typing import Any

import pandas as pd

from data_analysis_engine.integrations.meta_model_api import (
    MetaModelAPIClient,
)
from data_analysis_engine.orchestration.ai_plan_executor import (
    DeterministicAIPlanExecutor,
)
from data_analysis_engine.orchestration.dataset_binding import (
    DatasetPlanBinder,
)
from data_analysis_engine.orchestration.dataset_context import (
    build_safe_dataset_context,
)
from data_analysis_engine.orchestration.meta_interpreter import (
    MetaResultInterpreter,
)
from data_analysis_engine.orchestration.meta_planner import (
    MetaAnalysisPlanner,
)


class MetaAnalysisWorkflow:
    def __init__(
        self,
        client: MetaModelAPIClient,
    ) -> None:
        self.planner = MetaAnalysisPlanner(
            client
        )

        self.binder = DatasetPlanBinder()

        self.executor = (
            DeterministicAIPlanExecutor()
        )

        self.interpreter = (
            MetaResultInterpreter(
                client
            )
        )

    def run(
        self,
        *,
        dataframe: pd.DataFrame,
        user_request: str,
    ) -> dict[str, Any]:
        dataset_context = (
            build_safe_dataset_context(
                dataframe
            )
        )

        plan = self.planner.plan(
            user_request=user_request,
            dataset_context=(
                dataset_context
            ),
        )

        bound_plan = self.binder.bind(
            plan,
            dataframe,
        )

        execution = self.executor.execute(
            bound_plan
        )

        interpretation = (
            self.interpreter.interpret(
                user_request=user_request,
                dataset_context=(
                    dataset_context
                ),
                execution=execution,
            )
        )

        return {
            "dataset_context": (
                dataset_context
            ),
            "analysis_plan": (
                plan.as_dict()
            ),
            "execution": execution,
            "interpretation": (
                interpretation
            ),
        }
