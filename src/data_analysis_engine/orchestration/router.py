from data_analysis_engine.core.registry import MethodRegistry, registry
from data_analysis_engine.profiling.models import DatasetProfile
from data_analysis_engine.orchestration.models import (
    RejectedMethod,
    RoutingCandidate,
    RoutingReport,
)


class AnalyticalRouter:
    """
    Determines which registered analytical methods are
    compatible with a profiled dataset.

    It does not execute analyses.
    It only reasons about compatibility.
    """

    def __init__(
        self,
        method_registry: MethodRegistry | None = None,
    ) -> None:
        self.registry = method_registry or registry

    def route(
        self,
        profile: DatasetProfile,
    ) -> RoutingReport:
        candidates: list[RoutingCandidate] = []
        rejected: list[RejectedMethod] = []

        for method in self.registry.methods():
            compatible_columns = []

            for column in profile.columns:
                if column.role not in method.supported_roles:
                    continue

                if (
                    column.is_constant
                    and not method.allow_constant_columns
                ):
                    continue

                compatible_columns.append(column.name)

            if (
                len(compatible_columns)
                < method.minimum_compatible_columns
            ):
                rejected.append(
                    RejectedMethod(
                        method=method.name,
                        category=method.category,
                        reasons=[
                            (
                                "Not enough compatible columns. "
                                f"Required at least "
                                f"{method.minimum_compatible_columns}, "
                                f"found {len(compatible_columns)}."
                            )
                        ],
                    )
                )
                continue

            reasons = [
                (
                    "Compatible analytical role(s): "
                    + ", ".join(method.supported_roles)
                ),
                (
                    "Compatible column(s): "
                    + ", ".join(compatible_columns)
                ),
            ]

            candidates.append(
                RoutingCandidate(
                    method=method.name,
                    category=method.category,
                    selection_mode=method.selection_mode,
                    compatible_columns=compatible_columns,
                    reasons=reasons,
                )
            )

        return RoutingReport(
            dataset_fingerprint=profile.fingerprint,
            candidates=candidates,
            rejected=rejected,
        )
