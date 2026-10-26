from __future__ import annotations

import inspect
import re
from typing import Any, TYPE_CHECKING

from data_analysis_engine.core.bootstrap import (
    BUILTIN_METHODS,
)

if TYPE_CHECKING:
    from data_analysis_engine import (
        DataAnalysisEngine,
    )


_KWARGS_GET_PATTERN = re.compile(
    r"""kwargs\.get\(\s*["']([^"']+)["']"""
)


def _engine():
    from data_analysis_engine import (
        DataAnalysisEngine,
    )

    return DataAnalysisEngine()


def _parameter_keys_by_method() -> dict[
    str,
    list[str],
]:
    discovered = {}

    for method_class in BUILTIN_METHODS:
        method_name = getattr(
            method_class,
            "name",
            None,
        )

        if not method_name:
            continue

        try:
            source = inspect.getsource(
                method_class.execute
            )
        except (
            OSError,
            TypeError,
        ):
            discovered[
                str(method_name)
            ] = []
            continue

        discovered[
            str(method_name)
        ] = sorted(
            set(
                _KWARGS_GET_PATTERN.findall(
                    source
                )
            )
        )

    return discovered


def build_ai_orchestration_catalog(
    engine: "DataAnalysisEngine | None" = None,
) -> dict[str, Any]:
    engine = (
        engine
        if engine is not None
        else _engine()
    )

    registry = engine.available_methods()

    parameters = (
        _parameter_keys_by_method()
    )

    methods = []

    for method in registry:
        name = method[
            "name"
        ]

        methods.append(
            {
                "name": name,
                "category": method.get(
                    "category"
                ),
                "version": method.get(
                    "version"
                ),
                "description": method.get(
                    "description"
                ),
                "selection_mode": method.get(
                    "selection_mode"
                ),
                "analysis_unit": method.get(
                    "analysis_unit"
                ),
                "supported_roles": list(
                    method.get(
                        "supported_roles"
                    )
                    or []
                ),
                "minimum_compatible_columns": (
                    method.get(
                        "minimum_compatible_columns"
                    )
                ),
                "accepted_parameters": (
                    parameters.get(
                        name,
                        [],
                    )
                ),
            }
        )

    methods.sort(
        key=lambda item: item[
            "name"
        ]
    )

    return {
        "schema_version": "1.0",
        "engine_release": "core-v1",
        "native_method_count": len(
            methods
        ),
        "execution_policy": {
            "llm_may_select_methods": True,
            "llm_may_provide_parameters": True,
            "llm_may_interpret_results": True,
            "llm_may_compute_native_results": False,
            "all_native_calculations_require_engine_execution": True,
        },
        "methods": methods,
    }


def build_ai_analysis_plan_schema(
    engine: "DataAnalysisEngine | None" = None,
) -> dict[str, Any]:
    engine = (
        engine
        if engine is not None
        else _engine()
    )

    method_names = sorted(
        method[
            "name"
        ]
        for method
        in engine.available_methods()
    )

    return {
        "$schema": (
            "https://json-schema.org/draft/2020-12/schema"
        ),
        "title": (
            "Data Analysis Engine AI Plan"
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "plan_id",
            "calls",
        ],
        "properties": {
            "plan_id": {
                "type": "string",
                "minLength": 1,
            },
            "calls": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "call_id",
                        "method",
                        "parameters",
                    ],
                    "properties": {
                        "call_id": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "method": {
                            "type": "string",
                            "enum": method_names,
                        },
                        "parameters": {
                            "type": "object",
                        },
                        "bindings": {
                            "type": "object",
                            "additionalProperties": {
                                "oneOf": [
                                    {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    {
                                        "type": "array",
                                        "minItems": 1,
                                        "uniqueItems": True,
                                        "items": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                ]
                            },
                        },
                    },
                },
            },
        },
    }
