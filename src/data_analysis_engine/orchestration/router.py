from collections import Counter

from data_analysis_engine.core.registry import (
    MethodRegistry,
    registry,
)
from data_analysis_engine.orchestration.models import (
    RejectedMethod,
    RoutingCandidate,
    RoutingReport,
)
from data_analysis_engine.profiling.models import (
    DatasetProfile,
)


class AnalyticalRouter:
    """
    Determines which registered analytical methods are
    compatible with a profiled dataset.
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
            compatible_columns: list[str] = []
            compatible_roles: dict[str, str] = {}

            for column in profile.columns:
                if column.role not in method.supported_roles:
                    continue

                if (
                    column.is_constant
                    and not method.allow_constant_columns
                ):
                    continue

                compatible_columns.append(
                    column.name
                )

                compatible_roles[
                    column.name
                ] = column.role

            rejection_reasons: list[str] = []

            if (
                len(compatible_columns)
                < method.minimum_compatible_columns
            ):
                rejection_reasons.append(
                    (
                        "Not enough compatible columns. "
                        f"Required at least "
                        f"{method.minimum_compatible_columns}, "
                        f"found {len(compatible_columns)}."
                    )
                )

            requirements = dict(
                method.role_requirements
            )

            if requirements:
                counts = Counter(
                    compatible_roles.values()
                )

                for role, required_count in requirements.items():
                    available_count = counts.get(
                        role,
                        0,
                    )

                    if available_count < required_count:
                        rejection_reasons.append(
                            (
                                f"Role '{role}' requires "
                                f"{required_count} compatible "
                                f"column(s), found "
                                f"{available_count}."
                            )
                        )

            if rejection_reasons:
                rejected.append(
                    RejectedMethod(
                        method=method.name,
                        category=method.category,
                        reasons=rejection_reasons,
                    )
                )
                continue

            reasons = [
                (
                    "Compatible analytical role(s): "
                    + ", ".join(
                        method.supported_roles
                    )
                ),
                (
                    "Compatible column(s): "
                    + ", ".join(
                        compatible_columns
                    )
                ),
            ]

            if requirements:
                reasons.append(
                    "Required role structure satisfied: "
                    + ", ".join(
                        f"{role}={count}"
                        for role, count
                        in requirements.items()
                    )
                )

            candidates.append(
                RoutingCandidate(
                    method=method.name,
                    category=method.category,
                    selection_mode=method.selection_mode,
                    compatible_columns=compatible_columns,
                    compatible_column_roles=compatible_roles,
                    role_requirements=requirements,
                    reasons=reasons,
                )
            )

        return RoutingReport(
            dataset_fingerprint=profile.fingerprint,
            candidates=candidates,
            rejected=rejected,
        )
