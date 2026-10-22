from itertools import combinations
from typing import Iterable

from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.orchestration.models import (
    RoutingReport,
)
from data_analysis_engine.orchestration.planning_models import (
    AnalysisPlan,
    AnalysisTask,
)


class AnalysisPlanner:
    """
    Converts analytical routing recommendations into
    concrete executable tasks.

    Optionally restricts planning to explicitly selected methods.
    """

    def plan(
        self,
        routing_report: RoutingReport,
        *,
        methods: Iterable[str] | None = None,
    ) -> AnalysisPlan:
        selected_methods = self._normalize_methods(
            routing_report=routing_report,
            methods=methods,
        )

        tasks: list[AnalysisTask] = []

        for candidate in routing_report.candidates:
            if (
                selected_methods is not None
                and candidate.method not in selected_methods
            ):
                continue

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
                raise InvalidAnalysisInputError(
                    "Unknown analytical selection mode: "
                    f"'{candidate.selection_mode}'."
                )

        return AnalysisPlan(
            dataset_fingerprint=(
                routing_report.dataset_fingerprint
            ),
            tasks=tasks,
        )

    def _normalize_methods(
        self,
        *,
        routing_report: RoutingReport,
        methods: Iterable[str] | None,
    ) -> set[str] | None:
        if methods is None:
            return None

        if isinstance(methods, str):
            raise InvalidAnalysisInputError(
                "'methods' must be an iterable of method names, "
                "not a single string."
            )

        selected = set(methods)

        if not all(
            isinstance(method, str) and method.strip()
            for method in selected
        ):
            raise InvalidAnalysisInputError(
                "Every requested method name must be "
                "a non-empty string."
            )

        known_methods = {
            candidate.method
            for candidate in routing_report.candidates
        }

        known_methods.update(
            rejection.method
            for rejection in routing_report.rejected
        )

        unknown = (
            selected
            - known_methods
        )

        if unknown:
            raise InvalidAnalysisInputError(
                "Unknown analytical method(s): "
                + ", ".join(sorted(unknown))
            )

        return selected

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
            for left, right in combinations(
                columns,
                2,
            )
        ]
