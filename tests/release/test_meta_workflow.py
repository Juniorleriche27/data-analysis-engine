import json

import pandas as pd
import pytest

from data_analysis_engine import (
    DataAnalysisEngine,
)
from data_analysis_engine.integrations.meta_model_api import (
    MetaModelAPIClient,
)
from data_analysis_engine.orchestration.dataset_context import (
    build_safe_dataset_context,
)
from data_analysis_engine.orchestration.meta_interpreter import (
    MetaResultInterpreter,
)
from data_analysis_engine.orchestration.meta_planner import (
    MetaAnalysisPlanner,
)
from data_analysis_engine.orchestration.meta_workflow import (
    MetaAnalysisWorkflow,
)


def dummy_client(
    transport=None,
):
    return MetaModelAPIClient(
        api_key="test",
        transport=(
            transport
            or (
                lambda *args: {}
            )
        ),
    )


def test_safe_context_contains_no_raw_values():
    dataframe = pd.DataFrame(
        {
            "secret_sales": [
                123456.789,
                987654.321,
            ]
        }
    )

    context = (
        build_safe_dataset_context(
            dataframe
        )
    )

    serialized = json.dumps(
        context
    )

    assert (
        context[
            "contains_raw_values"
        ]
        is False
    )

    assert "123456.789" not in serialized
    assert "987654.321" not in serialized


def test_meta_planner_prompt_explains_bindings():
    planner = MetaAnalysisPlanner(
        dummy_client()
    )

    messages = planner.build_messages(
        user_request="Analyze sales.",
        dataset_context={
            "columns": [
                {
                    "name": "sales",
                }
            ]
        },
    )

    combined = " ".join(
        message["content"]
        for message
        in messages
    )

    assert "bindings" in combined

    assert (
        "column names only"
        in combined
    )


def test_meta_interpreter_returns_structured_result():
    def transport(
        url,
        headers,
        payload,
        timeout,
    ):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Median is 3.",
                                "key_findings": [
                                    "Median = 3.",
                                ],
                                "caveats": [],
                            }
                        )
                    }
                }
            ]
        }

    interpreter = MetaResultInterpreter(
        dummy_client(
            transport
        )
    )

    result = interpreter.interpret(
        user_request="Analyze sales.",
        dataset_context={
            "row_count": 5,
        },
        execution={
            "calls": [
                {
                    "result": {
                        "median": 3.0,
                    }
                }
            ]
        },
    )

    assert (
        result["summary"]
        == "Median is 3."
    )


def test_meta_workflow_runs_end_to_end_offline():
    def transport(
        url,
        headers,
        payload,
        timeout,
    ):
        schema_name = payload[
            "response_format"
        ][
            "json_schema"
        ][
            "name"
        ]

        if (
            schema_name
            == "DataAnalysisEnginePlan"
        ):
            content = {
                "plan_id": "p1",
                "calls": [
                    {
                        "call_id": "c1",
                        "method": (
                            "advanced_quantiles"
                        ),
                        "parameters": {
                            "probabilities": [
                                0.5,
                            ]
                        },
                        "bindings": {
                            "values": "sales",
                        },
                    }
                ],
            }

        else:
            content = {
                "summary": (
                    "The median is 3."
                ),
                "key_findings": [
                    "Median = 3.",
                ],
                "caveats": [],
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            content
                        )
                    }
                }
            ]
        }

    workflow = MetaAnalysisWorkflow(
        dummy_client(
            transport
        )
    )

    result = workflow.run(
        dataframe=pd.DataFrame(
            {
                "sales": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ]
            }
        ),
        user_request=(
            "What is the median?"
        ),
    )

    engine_result = result[
        "execution"
    ][
        "calls"
    ][
        0
    ][
        "result"
    ]

    assert engine_result[
        "median"
    ] == pytest.approx(
        3.0
    )

    assert result[
        "interpretation"
    ][
        "summary"
    ] == "The median is 3."


def test_meta_workflow_keeps_registry_at_95():
    assert len(
        DataAnalysisEngine()
        .available_methods()
    ) == 95
