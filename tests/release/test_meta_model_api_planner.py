import json

import pytest

from data_analysis_engine import (
    DataAnalysisEngine,
)
from data_analysis_engine.integrations.meta_model_api import (
    DEFAULT_META_BASE_URL,
    DEFAULT_META_MODEL,
    MetaModelAPIClient,
    MetaModelAPIError,
)
from data_analysis_engine.orchestration.meta_planner import (
    MetaAnalysisPlanner,
    MetaPlanningError,
)


def test_meta_client_uses_official_defaults():
    client = MetaModelAPIClient(
        api_key="test-key",
        transport=lambda *args: {},
    )

    assert (
        client.base_url
        == DEFAULT_META_BASE_URL
        == "https://api.meta.ai/v1"
    )

    assert (
        client.model
        == DEFAULT_META_MODEL
        == "muse-spark-1.3"
    )


def test_meta_client_requires_api_key(
    monkeypatch,
):
    monkeypatch.delenv(
        "MODEL_API_KEY",
        raising=False,
    )

    client = MetaModelAPIClient(
        transport=lambda *args: {},
    )

    with pytest.raises(
        MetaModelAPIError
    ):
        client.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ]
        )


def test_meta_client_builds_official_chat_endpoint():
    captured = {}

    def transport(
        url,
        headers,
        payload,
        timeout,
    ):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout"] = timeout

        return {
            "choices": [
                {
                    "message": {
                        "content": "ready",
                    }
                }
            ]
        }

    client = MetaModelAPIClient(
        api_key="secret-test-key",
        transport=transport,
    )

    client.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": "ping",
            }
        ]
    )

    assert captured[
        "url"
    ] == (
        "https://api.meta.ai/v1"
        "/chat/completions"
    )

    assert captured[
        "payload"
    ][
        "model"
    ] == "muse-spark-1.3"

    assert captured[
        "headers"
    ][
        "Authorization"
    ] == "Bearer secret-test-key"


def test_structured_completion_uses_json_schema():
    captured = {}

    def transport(
        url,
        headers,
        payload,
        timeout,
    ):
        captured.update(
            payload
        )

        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"value": 3}'
                        ),
                    }
                }
            ]
        }

    client = MetaModelAPIClient(
        api_key="test",
        transport=transport,
    )

    output = client.structured_completion(
        messages=[
            {
                "role": "user",
                "content": "test",
            }
        ],
        schema_name="Example",
        schema={
            "type": "object",
            "properties": {
                "value": {
                    "type": "integer",
                }
            },
            "required": [
                "value",
            ],
        },
    )

    assert output == {
        "value": 3,
    }

    assert captured[
        "response_format"
    ][
        "type"
    ] == "json_schema"

    assert captured[
        "response_format"
    ][
        "json_schema"
    ][
        "name"
    ] == "Example"


def test_meta_client_rejects_invalid_response_shape():
    client = MetaModelAPIClient(
        api_key="test",
        transport=lambda *args: {
            "unexpected": True,
        },
    )

    with pytest.raises(
        MetaModelAPIError
    ):
        client.structured_completion(
            messages=[
                {
                    "role": "user",
                    "content": "test",
                }
            ],
            schema_name="Example",
            schema={
                "type": "object",
            },
        )


def test_meta_client_rejects_non_json_structured_content():
    client = MetaModelAPIClient(
        api_key="test",
        transport=lambda *args: {
            "choices": [
                {
                    "message": {
                        "content": (
                            "not json"
                        ),
                    }
                }
            ]
        },
    )

    with pytest.raises(
        MetaModelAPIError
    ):
        client.structured_completion(
            messages=[
                {
                    "role": "user",
                    "content": "test",
                }
            ],
            schema_name="Example",
            schema={
                "type": "object",
            },
        )


def _planner_with_plan(
    plan,
    captured=None,
):
    def transport(
        url,
        headers,
        payload,
        timeout,
    ):
        if captured is not None:
            captured[
                "payload"
            ] = payload

        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            plan
                        ),
                    }
                }
            ]
        }

    return MetaAnalysisPlanner(
        MetaModelAPIClient(
            api_key="test",
            transport=transport,
        )
    )


def test_meta_planner_accepts_valid_native_plan():
    planner = _planner_with_plan(
        {
            "plan_id": "plan-1",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": (
                        "advanced_quantiles"
                    ),
                    "parameters": {
                        "probabilities": [
                            0.25,
                            0.5,
                            0.75,
                        ]
                    },
                }
            ],
        }
    )

    plan = planner.plan(
        user_request=(
            "Give me robust quantiles "
            "for the sales column."
        ),
        dataset_context={
            "columns": [
                {
                    "name": "sales",
                    "role": "measure",
                }
            ]
        },
    )

    assert (
        plan.calls[
            0
        ].method
        == "advanced_quantiles"
    )


def test_meta_planner_rejects_unknown_method():
    planner = _planner_with_plan(
        {
            "plan_id": "bad",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": (
                        "invented_ai_math"
                    ),
                    "parameters": {},
                }
            ],
        }
    )

    with pytest.raises(
        MetaPlanningError
    ):
        planner.plan(
            user_request="Analyze sales.",
            dataset_context={
                "columns": [
                    {
                        "name": "sales",
                        "role": "measure",
                    }
                ]
            },
        )


def test_meta_planner_forbids_raw_data_parameter():
    planner = _planner_with_plan(
        {
            "plan_id": "bad",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": (
                        "ordinary_least_squares"
                    ),
                    "parameters": {
                        "target_column": "y",
                        "data": {
                            "x": [
                                1,
                                2,
                            ],
                            "y": [
                                3,
                                4,
                            ],
                        },
                    },
                }
            ],
        }
    )

    with pytest.raises(
        MetaPlanningError
    ):
        planner.plan(
            user_request=(
                "Regress y on x."
            ),
            dataset_context={
                "columns": [
                    {
                        "name": "x",
                        "role": "measure",
                    },
                    {
                        "name": "y",
                        "role": "measure",
                    },
                ]
            },
        )


def test_meta_planner_forbids_values_parameter():
    planner = _planner_with_plan(
        {
            "plan_id": "bad",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": (
                        "advanced_quantiles"
                    ),
                    "parameters": {
                        "values": [
                            1,
                            2,
                            3,
                        ]
                    },
                }
            ],
        }
    )

    with pytest.raises(
        MetaPlanningError
    ):
        planner.plan(
            user_request=(
                "Analyze sales."
            ),
            dataset_context={
                "columns": [
                    {
                        "name": "sales",
                        "role": "measure",
                    }
                ]
            },
        )


def test_meta_planner_prompt_contains_request_and_dataset_context():
    captured = {}

    planner = _planner_with_plan(
        {
            "plan_id": "plan-1",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": (
                        "advanced_quantiles"
                    ),
                    "parameters": {},
                }
            ],
        },
        captured=captured,
    )

    planner.plan(
        user_request=(
            "Study the revenue distribution."
        ),
        dataset_context={
            "columns": [
                {
                    "name": "revenue",
                    "role": "measure",
                }
            ]
        },
    )

    messages = captured[
        "payload"
    ][
        "messages"
    ]

    combined = " ".join(
        message[
            "content"
        ]
        for message
        in messages
    )

    assert (
        "Study the revenue distribution."
        in combined
    )

    assert "revenue" in combined

    assert (
        "MUST NOT perform calculations"
        in combined
    )


def test_meta_integration_does_not_change_native_registry():
    methods = (
        DataAnalysisEngine()
        .available_methods()
    )

    assert len(
        methods
    ) == 95

    assert len(
        {
            method[
                "name"
            ]
            for method
            in methods
        }
    ) == 95
