from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from data_analysis_engine.integrations.meta_model_api import (
    MetaModelAPIClient,
)


class MetaInterpretationError(
    RuntimeError
):
    pass


class MetaResultInterpreter:
    def __init__(
        self,
        client: MetaModelAPIClient,
    ) -> None:
        self.client = client

    @staticmethod
    def schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "summary",
                "key_findings",
                "caveats",
            ],
            "properties": {
                "summary": {
                    "type": "string",
                },
                "key_findings": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
                "caveats": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                },
            },
        }

    def interpret(
        self,
        *,
        user_request: str,
        dataset_context: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> dict[str, Any]:
        if (
            not isinstance(
                user_request,
                str,
            )
            or not user_request.strip()
        ):
            raise MetaInterpretationError(
                "User request is required."
            )

        payload = {
            "user_request": (
                user_request.strip()
            ),
            "dataset_context": dict(
                dataset_context
            ),
            "deterministic_engine_execution": dict(
                execution
            ),
        }

        return self.client.structured_completion(
            messages=[
                {
                    "role": "developer",
                    "content": (
                        "Interpret only the supplied results "
                        "from the deterministic engine. "
                        "Do not recalculate statistics. "
                        "Do not alter numerical results. "
                        "Do not invent missing results. "
                        "Explain findings and caveats clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                },
            ],
            schema_name=(
                "DataAnalysisInterpretation"
            ),
            schema=self.schema(),
        )
