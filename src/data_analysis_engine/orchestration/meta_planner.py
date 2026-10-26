from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from data_analysis_engine.integrations.meta_model_api import (
    MetaModelAPIClient,
)
from data_analysis_engine.orchestration.ai_catalog import (
    build_ai_analysis_plan_schema,
    build_ai_orchestration_catalog,
)
from data_analysis_engine.orchestration.ai_contracts import (
    AIAnalysisPlan,
    AIOrchestrationContractError,
)


class MetaPlanningError(
    RuntimeError
):
    """Raised when a Meta-produced analysis plan is unsafe."""


_FORBIDDEN_DATA_PARAMETERS = {
    "data",
    "values",
}


class MetaAnalysisPlanner:
    """
    Muse Spark selects native analytical methods.

    It does not receive authority to calculate results
    and it does not inject dataset values into the plan.
    """

    def __init__(
        self,
        client: MetaModelAPIClient,
    ) -> None:
        self.client = client

    @staticmethod
    def _planning_catalog() -> list[
        dict[str, Any]
    ]:
        catalog = (
            build_ai_orchestration_catalog()
        )

        return [
            {
                "name": method[
                    "name"
                ],
                "category": method[
                    "category"
                ],
                "description": method[
                    "description"
                ],
                "selection_mode": method[
                    "selection_mode"
                ],
                "analysis_unit": method[
                    "analysis_unit"
                ],
                "supported_roles": method[
                    "supported_roles"
                ],
                "accepted_parameters": method[
                    "accepted_parameters"
                ],
            }
            for method
            in catalog[
                "methods"
            ]
        ]

    def build_messages(
        self,
        *,
        user_request: str,
        dataset_context: Mapping[
            str,
            Any,
        ],
    ) -> list[dict[str, str]]:
        if (
            not isinstance(
                user_request,
                str,
            )
            or not user_request.strip()
        ):
            raise MetaPlanningError(
                "'user_request' must be "
                "a non-empty string."
            )

        if not isinstance(
            dataset_context,
            Mapping,
        ):
            raise MetaPlanningError(
                "'dataset_context' must be an object."
            )

        catalog = (
            self._planning_catalog()
        )

        developer_message = {
            "role": "developer",
            "content": (
                "You are the planning layer for a "
                "deterministic data-analysis engine. "
                "You MUST NOT perform calculations. "
                "You MUST NOT invent analytical results. "
                "You MUST NOT include raw dataset values. "
                "Select only methods from the supplied "
                "native method catalog. "
                "Return only an analysis plan matching "
                "the supplied JSON schema. "
                "Parameters may contain configuration, "
                "column names, thresholds, strategy names, "
                "target_column, predictor_columns, columns, "
                "lags, confidence levels, and similar "
                "method configuration. "
                "Do not use parameter keys named 'data' "
                "or 'values'; the application will bind "
                "the real dataset after planning. "
                "Use the separate bindings object to map "
                "native data inputs to dataset column names. "
                "Examples: bindings={values: sales} for a "
                "single-column method, or bindings={data: "
                "[x, y]} for a multi-column method. "
                "Bindings contain column names only, never "
                "raw dataset values."
            ),
        }

        request_payload = {
            "user_request": (
                user_request.strip()
            ),
            "dataset_context": dict(
                dataset_context
            ),
            "native_method_catalog": (
                catalog
            ),
        }

        user_message = {
            "role": "user",
            "content": json.dumps(
                request_payload,
                ensure_ascii=False,
            ),
        }

        return [
            developer_message,
            user_message,
        ]

    @staticmethod
    def _validate_no_dataset_fabrication(
        plan: AIAnalysisPlan,
    ) -> None:
        for call in plan.calls:
            forbidden = (
                set(
                    call.parameters.keys()
                )
                & _FORBIDDEN_DATA_PARAMETERS
            )

            if forbidden:
                raise MetaPlanningError(
                    "Meta plan attempted to provide "
                    "dataset values directly using: "
                    + ", ".join(
                        sorted(
                            forbidden
                        )
                    )
                )

    def plan(
        self,
        *,
        user_request: str,
        dataset_context: Mapping[
            str,
            Any,
        ],
    ) -> AIAnalysisPlan:
        messages = self.build_messages(
            user_request=(
                user_request
            ),
            dataset_context=(
                dataset_context
            ),
        )

        schema = (
            build_ai_analysis_plan_schema()
        )

        raw_plan = (
            self.client.structured_completion(
                messages=messages,
                schema_name=(
                    "DataAnalysisEnginePlan"
                ),
                schema=schema,
            )
        )

        registered_methods = {
            method[
                "name"
            ]
            for method
            in build_ai_orchestration_catalog()[
                "methods"
            ]
        }

        try:
            plan = (
                AIAnalysisPlan.from_mapping(
                    raw_plan,
                    registered_methods=(
                        registered_methods
                    ),
                )
            )

        except AIOrchestrationContractError as exc:
            raise MetaPlanningError(
                "Meta returned an invalid "
                "analysis plan."
            ) from exc

        self._validate_no_dataset_fabrication(
            plan
        )

        return plan
