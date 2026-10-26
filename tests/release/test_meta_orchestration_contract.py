import json
from pathlib import Path

import pytest

from data_analysis_engine import (
    DataAnalysisEngine,
)
from data_analysis_engine.orchestration.ai_catalog import (
    build_ai_analysis_plan_schema,
    build_ai_orchestration_catalog,
)
from data_analysis_engine.orchestration.ai_contracts import (
    AIAnalysisPlan,
    AIOrchestrationContractError,
)
from data_analysis_engine.orchestration.ai_plan_executor import (
    DeterministicAIPlanExecutor,
)


ROOT = Path(
    __file__
).resolve().parents[2]


def registered_methods():
    return {
        method["name"]
        for method
        in DataAnalysisEngine().available_methods()
    }


def test_ai_catalog_contains_exactly_95_methods():
    catalog = build_ai_orchestration_catalog()

    assert catalog[
        "native_method_count"
    ] == 95

    assert len(
        catalog[
            "methods"
        ]
    ) == 95


def test_ai_catalog_matches_core_v1_manifest():
    manifest = json.loads(
        (
            ROOT
            / "docs"
            / "native_methods_manifest.json"
        ).read_text()
    )

    manifest_names = [
        method["name"]
        for method
        in manifest["methods"]
    ]

    catalog_names = [
        method["name"]
        for method
        in build_ai_orchestration_catalog()[
            "methods"
        ]
    ]

    assert catalog_names == manifest_names


def test_ai_catalog_forbids_llm_native_calculation():
    policy = (
        build_ai_orchestration_catalog()[
            "execution_policy"
        ]
    )

    assert (
        policy[
            "llm_may_compute_native_results"
        ]
        is False
    )

    assert (
        policy[
            "all_native_calculations_require_engine_execution"
        ]
        is True
    )


def test_ai_schema_contains_only_registered_methods():
    schema = (
        build_ai_analysis_plan_schema()
    )

    schema_methods = set(
        schema[
            "properties"
        ][
            "calls"
        ][
            "items"
        ][
            "properties"
        ][
            "method"
        ][
            "enum"
        ]
    )

    assert (
        schema_methods
        == registered_methods()
    )


def test_valid_ai_plan_is_accepted():
    plan = AIAnalysisPlan.from_mapping(
        {
            "plan_id": "plan-1",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": "advanced_quantiles",
                    "parameters": {
                        "values": [
                            1.0,
                            2.0,
                            3.0,
                        ]
                    },
                }
            ],
        },
        registered_methods=registered_methods(),
    )

    assert plan.plan_id == "plan-1"

    assert (
        plan.calls[0].method
        == "advanced_quantiles"
    )


def test_unknown_ai_method_is_rejected():
    with pytest.raises(
        AIOrchestrationContractError
    ):
        AIAnalysisPlan.from_mapping(
            {
                "plan_id": "bad",
                "calls": [
                    {
                        "call_id": "call-1",
                        "method": "llm_magic_math",
                        "parameters": {},
                    }
                ],
            },
            registered_methods=registered_methods(),
        )


def test_ai_cannot_attach_analytical_result():
    with pytest.raises(
        AIOrchestrationContractError
    ):
        AIAnalysisPlan.from_mapping(
            {
                "plan_id": "bad",
                "calls": [
                    {
                        "call_id": "call-1",
                        "method": "advanced_quantiles",
                        "parameters": {},
                        "result": {
                            "median": 999,
                        },
                    }
                ],
            },
            registered_methods=registered_methods(),
        )


def test_ai_cannot_inject_computed_result_parameter():
    executor = (
        DeterministicAIPlanExecutor()
    )

    with pytest.raises(
        AIOrchestrationContractError
    ):
        executor.validate_plan(
            {
                "plan_id": "bad",
                "calls": [
                    {
                        "call_id": "call-1",
                        "method": "advanced_quantiles",
                        "parameters": {
                            "values": [
                                1.0,
                                2.0,
                                3.0,
                            ],
                            "computed_result": {
                                "median": 999,
                            },
                        },
                    }
                ],
            }
        )


def test_duplicate_ai_call_ids_are_rejected():
    executor = (
        DeterministicAIPlanExecutor()
    )

    with pytest.raises(
        AIOrchestrationContractError
    ):
        executor.validate_plan(
            {
                "plan_id": "bad",
                "calls": [
                    {
                        "call_id": "same",
                        "method": "advanced_quantiles",
                        "parameters": {
                            "values": [
                                1.0,
                                2.0,
                                3.0,
                            ]
                        },
                    },
                    {
                        "call_id": "same",
                        "method": "distribution_shape_statistics",
                        "parameters": {
                            "values": [
                                1.0,
                                2.0,
                                3.0,
                            ]
                        },
                    },
                ],
            }
        )


def test_ai_executor_delegates_to_native_engine():
    executor = (
        DeterministicAIPlanExecutor()
    )

    output = executor.execute(
        {
            "plan_id": "quantile-plan",
            "calls": [
                {
                    "call_id": "quantiles",
                    "method": "advanced_quantiles",
                    "parameters": {
                        "values": [
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                            5.0,
                        ]
                    },
                }
            ],
        }
    )

    assert output[
        "execution_policy"
    ] == "deterministic_engine_only"

    assert output[
        "call_count"
    ] == 1

    assert output[
        "calls"
    ][
        0
    ][
        "result"
    ][
        "median"
    ] == pytest.approx(
        3.0
    )

    assert len(
        DataAnalysisEngine()
        .available_methods()
    ) == 95
