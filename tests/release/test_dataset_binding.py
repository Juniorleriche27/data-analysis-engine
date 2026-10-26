import pandas as pd
import pytest

from data_analysis_engine import (
    DataAnalysisEngine,
)
from data_analysis_engine.orchestration.ai_contracts import (
    AIAnalysisPlan,
)
from data_analysis_engine.orchestration.dataset_binding import (
    DatasetBindingError,
    DatasetPlanBinder,
)


def registered_methods():
    return {
        method["name"]
        for method
        in DataAnalysisEngine().available_methods()
    }


def make_plan(
    method,
    parameters,
    bindings,
):
    return AIAnalysisPlan.from_mapping(
        {
            "plan_id": "plan-1",
            "calls": [
                {
                    "call_id": "call-1",
                    "method": method,
                    "parameters": parameters,
                    "bindings": bindings,
                }
            ],
        },
        registered_methods=registered_methods(),
    )


def test_binder_binds_single_column_to_values():
    dataframe = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
                3.0,
            ]
        }
    )

    plan = make_plan(
        "advanced_quantiles",
        {
            "probabilities": [
                0.5,
            ]
        },
        {
            "values": "sales",
        },
    )

    bound = DatasetPlanBinder().bind(
        plan,
        dataframe,
    )

    assert bound.calls[0].parameters[
        "values"
    ] == [
        1.0,
        2.0,
        3.0,
    ]

    assert bound.calls[0].bindings == {}


def test_binder_binds_multiple_columns_to_data():
    dataframe = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
            ],
            "y": [
                3.0,
                4.0,
            ],
        }
    )

    plan = make_plan(
        "ordinary_least_squares",
        {
            "target_column": "y",
            "predictor_columns": [
                "x",
            ],
        },
        {
            "data": [
                "x",
                "y",
            ],
        },
    )

    bound = DatasetPlanBinder().bind(
        plan,
        dataframe,
    )

    assert bound.calls[0].parameters[
        "data"
    ] == {
        "x": [
            1.0,
            2.0,
        ],
        "y": [
            3.0,
            4.0,
        ],
    }


def test_binder_preserves_configuration_parameters():
    dataframe = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
                3.0,
            ]
        }
    )

    plan = make_plan(
        "advanced_quantiles",
        {
            "probabilities": [
                0.25,
                0.5,
                0.75,
            ]
        },
        {
            "values": "sales",
        },
    )

    bound = DatasetPlanBinder().bind(
        plan,
        dataframe,
    )

    assert bound.calls[0].parameters[
        "probabilities"
    ] == [
        0.25,
        0.5,
        0.75,
    ]


def test_binder_rejects_unknown_column():
    dataframe = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
            ]
        }
    )

    plan = make_plan(
        "advanced_quantiles",
        {},
        {
            "values": "missing_column",
        },
    )

    with pytest.raises(
        DatasetBindingError
    ):
        DatasetPlanBinder().bind(
            plan,
            dataframe,
        )


def test_binder_rejects_missing_bindings():
    dataframe = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
            ]
        }
    )

    plan = make_plan(
        "advanced_quantiles",
        {},
        {},
    )

    with pytest.raises(
        DatasetBindingError
    ):
        DatasetPlanBinder().bind(
            plan,
            dataframe,
        )


def test_binder_rejects_unknown_native_parameter():
    dataframe = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
            ]
        }
    )

    plan = make_plan(
        "advanced_quantiles",
        {},
        {
            "fake_input": "sales",
        },
    )

    with pytest.raises(
        DatasetBindingError
    ):
        DatasetPlanBinder().bind(
            plan,
            dataframe,
        )


def test_binder_rejects_parameter_binding_conflict():
    dataframe = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
            ]
        }
    )

    plan = make_plan(
        "advanced_quantiles",
        {
            "values": [
                999.0,
            ]
        },
        {
            "values": "sales",
        },
    )

    with pytest.raises(
        DatasetBindingError
    ):
        DatasetPlanBinder().bind(
            plan,
            dataframe,
        )
