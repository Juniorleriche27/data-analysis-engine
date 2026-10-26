from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


_RESERVED_RESULT_KEYS = {
    "result",
    "results",
    "engine_result",
    "computed_result",
    "calculated_result",
    "final_answer",
}


class AIOrchestrationContractError(
    ValueError
):
    """Raised when an AI plan violates the engine contract."""


def _assert_no_result_injection(
    value: Any,
    *,
    path: str,
) -> None:
    if isinstance(
        value,
        Mapping,
    ):
        for key, nested_value in value.items():
            key_text = str(
                key
            )

            if (
                key_text.lower()
                in _RESERVED_RESULT_KEYS
            ):
                raise AIOrchestrationContractError(
                    "AI plans cannot provide analytical "
                    f"results directly: reserved key "
                    f"'{key_text}' found at {path}."
                )

            _assert_no_result_injection(
                nested_value,
                path=(
                    f"{path}.{key_text}"
                ),
            )

    elif isinstance(
        value,
        (list, tuple),
    ):
        for index, nested_value in enumerate(
            value
        ):
            _assert_no_result_injection(
                nested_value,
                path=(
                    f"{path}[{index}]"
                ),
            )


def _validate_bindings(
    value: Any,
) -> dict[str, Any]:
    if value is None:
        return {}

    if not isinstance(
        value,
        Mapping,
    ):
        raise AIOrchestrationContractError(
            "'bindings' must be an object."
        )

    normalized: dict[
        str,
        Any,
    ] = {}

    for parameter_name, source in value.items():
        if (
            not isinstance(
                parameter_name,
                str,
            )
            or not parameter_name.strip()
        ):
            raise AIOrchestrationContractError(
                "Binding parameter names must "
                "be non-empty strings."
            )

        parameter_name = (
            parameter_name.strip()
        )

        if isinstance(
            source,
            str,
        ):
            if not source.strip():
                raise AIOrchestrationContractError(
                    "Bound column names must "
                    "be non-empty strings."
                )

            normalized[
                parameter_name
            ] = source.strip()

        elif isinstance(
            source,
            (list, tuple),
        ):
            if not source:
                raise AIOrchestrationContractError(
                    "Column-list bindings "
                    "cannot be empty."
                )

            columns: list[str] = []

            for column in source:
                if (
                    not isinstance(
                        column,
                        str,
                    )
                    or not column.strip()
                ):
                    raise AIOrchestrationContractError(
                        "Every bound column must "
                        "be a non-empty string."
                    )

                columns.append(
                    column.strip()
                )

            if len(
                columns
            ) != len(
                set(
                    columns
                )
            ):
                raise AIOrchestrationContractError(
                    "Bindings cannot contain "
                    "duplicate columns."
                )

            normalized[
                parameter_name
            ] = columns

        else:
            raise AIOrchestrationContractError(
                "A binding source must be a column "
                "name or a list of column names."
            )

    return normalized


@dataclass(
    frozen=True
)
class AIAnalysisCall:
    call_id: str
    method: str
    parameters: dict[str, Any]
    bindings: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        registered_methods: set[str],
    ) -> "AIAnalysisCall":
        if not isinstance(
            payload,
            Mapping,
        ):
            raise AIOrchestrationContractError(
                "Each analysis call must be an object."
            )

        allowed_keys = {
            "call_id",
            "method",
            "parameters",
            "bindings",
        }

        unknown_keys = (
            set(
                payload.keys()
            )
            - allowed_keys
        )

        if unknown_keys:
            raise AIOrchestrationContractError(
                "Unsupported analysis-call fields: "
                + ", ".join(
                    sorted(
                        str(key)
                        for key
                        in unknown_keys
                    )
                )
            )

        call_id = payload.get(
            "call_id"
        )

        if (
            not isinstance(
                call_id,
                str,
            )
            or not call_id.strip()
        ):
            raise AIOrchestrationContractError(
                "'call_id' must be a non-empty string."
            )

        method = payload.get(
            "method"
        )

        if (
            not isinstance(
                method,
                str,
            )
            or not method.strip()
        ):
            raise AIOrchestrationContractError(
                "'method' must be a non-empty string."
            )

        method = method.strip()

        if method not in registered_methods:
            raise AIOrchestrationContractError(
                f"Unknown analytical method: '{method}'."
            )

        parameters = payload.get(
            "parameters",
            {},
        )

        if not isinstance(
            parameters,
            Mapping,
        ):
            raise AIOrchestrationContractError(
                "'parameters' must be an object."
            )

        normalized_parameters = dict(
            parameters
        )

        _assert_no_result_injection(
            normalized_parameters,
            path="parameters",
        )

        bindings = _validate_bindings(
            payload.get(
                "bindings",
                {},
            )
        )

        return cls(
            call_id=call_id.strip(),
            method=method,
            parameters=normalized_parameters,
            bindings=bindings,
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "method": self.method,
            "parameters": dict(
                self.parameters
            ),
            "bindings": dict(
                self.bindings
            ),
        }


@dataclass(
    frozen=True
)
class AIAnalysisPlan:
    plan_id: str
    calls: tuple[
        AIAnalysisCall,
        ...
    ]

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        registered_methods: set[str],
    ) -> "AIAnalysisPlan":
        if not isinstance(
            payload,
            Mapping,
        ):
            raise AIOrchestrationContractError(
                "Analysis plan must be an object."
            )

        allowed_keys = {
            "plan_id",
            "calls",
        }

        unknown_keys = (
            set(
                payload.keys()
            )
            - allowed_keys
        )

        if unknown_keys:
            raise AIOrchestrationContractError(
                "Unsupported analysis-plan fields: "
                + ", ".join(
                    sorted(
                        str(key)
                        for key
                        in unknown_keys
                    )
                )
            )

        plan_id = payload.get(
            "plan_id"
        )

        if (
            not isinstance(
                plan_id,
                str,
            )
            or not plan_id.strip()
        ):
            raise AIOrchestrationContractError(
                "'plan_id' must be a non-empty string."
            )

        raw_calls = payload.get(
            "calls"
        )

        if (
            not isinstance(
                raw_calls,
                (list, tuple),
            )
            or not raw_calls
        ):
            raise AIOrchestrationContractError(
                "'calls' must contain at least "
                "one analysis call."
            )

        calls = tuple(
            AIAnalysisCall.from_mapping(
                call,
                registered_methods=(
                    registered_methods
                ),
            )
            for call
            in raw_calls
        )

        call_ids = [
            call.call_id
            for call
            in calls
        ]

        if len(
            call_ids
        ) != len(
            set(
                call_ids
            )
        ):
            raise AIOrchestrationContractError(
                "Analysis call IDs must be unique."
            )

        return cls(
            plan_id=plan_id.strip(),
            calls=calls,
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "calls": [
                call.as_dict()
                for call
                in self.calls
            ],
        }
