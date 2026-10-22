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
                    AnalysisTask(
                        method=candidate.method,
                        category=candidate.category,
                        selection_mode="per_column",
                        columns=[column],
                    )
                    for column
                    in candidate.compatible_columns
                )

            elif candidate.selection_mode == "time_axis":
                time_columns = [
                    column
                    for column
                    in candidate.compatible_columns
                    if candidate.compatible_column_roles[
                        column
                    ] == "time"
                ]

                tasks.extend(
                    AnalysisTask(
                        method=candidate.method,
                        category=candidate.category,
                        selection_mode="time_axis",
                        columns=[column],
                    )
                    for column in time_columns
                )

            elif candidate.selection_mode == "pairwise":
                tasks.extend(
                    AnalysisTask(
                        method=candidate.method,
                        category=candidate.category,
                        selection_mode="pairwise",
                        columns=[left, right],
                    )
                    for left, right in combinations(
                        candidate.compatible_columns,
                        2,
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

            elif candidate.selection_mode == "time_series_pairwise":
                time_columns = [
                    column
                    for column
                    in candidate.compatible_columns
                    if candidate.compatible_column_roles[
                        column
                    ] == "time"
                ]

                measure_columns = [
                    column
                    for column
                    in candidate.compatible_columns
                    if candidate.compatible_column_roles[
                        column
                    ] == "measure"
                ]

                for time_column in time_columns:
                    for left, right in combinations(
                        measure_columns,
                        2,
                    ):
                        tasks.append(
                            AnalysisTask(
                                method=candidate.method,
                                category=candidate.category,
                                selection_mode="time_series_pairwise",
                                columns=[
                                    time_column,
                                    left,
                                    right,
                                ],
                            )
                        )

            elif candidate.selection_mode == "time_series":
                time_columns = [
                    column
                    for column
                    in candidate.compatible_columns
                    if candidate.compatible_column_roles[
                        column
                    ] == "time"
                ]

                measure_columns = [
                    column
                    for column
                    in candidate.compatible_columns
                    if candidate.compatible_column_roles[
                        column
                    ] == "measure"
                ]

                for time_column in time_columns:
                    for measure_column in measure_columns:
                        tasks.append(
                            AnalysisTask(
                                method=candidate.method,
                                category=candidate.category,
                                selection_mode="time_series",
                                columns=[
                                    time_column,
                                    measure_column,
                                ],
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
            isinstance(method, str)
            and method.strip()
            for method in selected
        ):
            raise InvalidAnalysisInputError(
                "Every requested method name must be "
                "a non-empty string."
            )

        known_methods = {
            candidate.method
            for candidate
            in routing_report.candidates
        }

        known_methods.update(
            rejection.method
            for rejection
            in routing_report.rejected
        )

        unknown = selected - known_methods

        if unknown:
            raise InvalidAnalysisInputError(
                "Unknown analytical method(s): "
                + ", ".join(
                    sorted(unknown)
                )
            )

        return selected
