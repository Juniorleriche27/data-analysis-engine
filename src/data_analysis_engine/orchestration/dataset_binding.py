from __future__ import annotations

import pandas as pd

from data_analysis_engine.orchestration.ai_catalog import (
    build_ai_orchestration_catalog,
)
from data_analysis_engine.orchestration.ai_contracts import (
    AIAnalysisCall,
    AIAnalysisPlan,
)


class DatasetBindingError(
    ValueError
):
    """Raised when an AI plan cannot bind safely to a dataset."""


class DatasetPlanBinder:
    def __init__(
        self,
    ) -> None:
        catalog = (
            build_ai_orchestration_catalog()
        )

        self.methods = {
            method["name"]: method
            for method
            in catalog["methods"]
        }

    @staticmethod
    def _validate_column(
        dataframe: pd.DataFrame,
        column: str,
    ) -> None:
        if column not in dataframe.columns:
            raise DatasetBindingError(
                f"Dataset column '{column}' does not exist."
            )

    def bind(
        self,
        plan: AIAnalysisPlan,
        dataframe: pd.DataFrame,
    ) -> AIAnalysisPlan:
        if not isinstance(
            plan,
            AIAnalysisPlan,
        ):
            raise DatasetBindingError(
                "'plan' must be an AIAnalysisPlan."
            )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise DatasetBindingError(
                "'dataframe' must be a pandas DataFrame."
            )

        bound_calls = []

        for call in plan.calls:
            metadata = self.methods.get(
                call.method
            )

            if metadata is None:
                raise DatasetBindingError(
                    f"Unknown method '{call.method}'."
                )

            if not call.bindings:
                raise DatasetBindingError(
                    f"Call '{call.call_id}' has no bindings."
                )

            accepted_parameters = set(
                metadata[
                    "accepted_parameters"
                ]
            )

            parameters = dict(
                call.parameters
            )

            for (
                parameter_name,
                source,
            ) in call.bindings.items():

                if (
                    parameter_name
                    not in accepted_parameters
                ):
                    raise DatasetBindingError(
                        f"Method '{call.method}' does not "
                        f"accept bound parameter "
                        f"'{parameter_name}'."
                    )

                if parameter_name in parameters:
                    raise DatasetBindingError(
                        f"Parameter '{parameter_name}' cannot "
                        "be both configured and bound."
                    )

                if isinstance(
                    source,
                    str,
                ):
                    self._validate_column(
                        dataframe,
                        source,
                    )

                    parameters[
                        parameter_name
                    ] = dataframe[
                        source
                    ].tolist()

                elif isinstance(
                    source,
                    list,
                ):
                    if parameter_name != "data":
                        raise DatasetBindingError(
                            "Multi-column bindings are only "
                            "allowed for the 'data' parameter."
                        )

                    for column in source:
                        self._validate_column(
                            dataframe,
                            column,
                        )

                    parameters[
                        "data"
                    ] = {
                        column: dataframe[
                            column
                        ].tolist()
                        for column
                        in source
                    }

                else:
                    raise DatasetBindingError(
                        "Unsupported binding source."
                    )

            bound_calls.append(
                AIAnalysisCall(
                    call_id=call.call_id,
                    method=call.method,
                    parameters=parameters,
                    bindings={},
                )
            )

        return AIAnalysisPlan(
            plan_id=plan.plan_id,
            calls=tuple(
                bound_calls
            ),
        )
