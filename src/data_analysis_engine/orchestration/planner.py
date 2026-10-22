from itertools import combinations

from data_analysis_engine.orchestration.models import RoutingReport
from data_analysis_engine.orchestration.planning_models import (
    AnalysisPlan,
    AnalysisTask,
)


class AnalysisPlanner:
    """
    Converts analytical routing recommendations into
    concrete executable tasks.

    The planner is method-agnostic. It relies only on
    each method's declared selection mode.
    """

    def plan(
        self,
        routing_report: RoutingReport,
    ) -> AnalysisPlan:
        tasks: list[AnalysisTask] = []

        for candidate in routing_report.candidates:
            if candidate.selection_mode == "per_column":
                tasks.extend(
                    self._build_per_column_tasks(
                        candidate.method,
                        candidate.category,
                        candidate.compatible_columns,
                    )
                )

            elif candidate.selection_mode == "pairwise":
                tasks.extend(
                    self._build_pairwise_tasks(
                        candidate.method,
                        candidate.category,
                        candidate.compatible_columns,
                    )
                )

            elif candidate.selection_mode == "joint":
                tasks.append(
                    AnalysisTask(
                        method=candidate.method,
                        category=candidate.category,
                        selection_mode="joint",
                        columns=list(
                            candidate.compatible_columns
                        ),
                    )
                )

            else:
                raise ValueError(
                    "Unknown analytical selection mode: "
                    f"'{candidate.selection_mode}'."
                )

        return AnalysisPlan(
            dataset_fingerprint=(
                routing_report.dataset_fingerprint
            ),
            tasks=tasks,
        )

    def _build_per_column_tasks(
        self,
        method: str,
        category: str,
        columns: list[str],
    ) -> list[AnalysisTask]:
        return [
            AnalysisTask(
                method=method,
                category=category,
                selection_mode="per_column",
                columns=[column],
            )
            for column in columns
        ]

    def _build_pairwise_tasks(
        self,
        method: str,
        category: str,
        columns: list[str],
    ) -> list[AnalysisTask]:
        return [
            AnalysisTask(
                method=method,
                category=category,
                selection_mode="pairwise",
                columns=[left, right],
            )
            for left, right in combinations(columns, 2)
        ]
