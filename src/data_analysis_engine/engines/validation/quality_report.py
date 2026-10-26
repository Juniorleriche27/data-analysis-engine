from typing import Any, Callable

import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)
from data_analysis_engine.engines.validation.cardinality import (
    CardinalityDiagnostic,
)
from data_analysis_engine.engines.validation.constantness import (
    ConstantColumnDiagnostic,
)
from data_analysis_engine.engines.validation.duplicate_rows import (
    DuplicateRowDiagnostic,
)
from data_analysis_engine.engines.validation.missingness import (
    MissingValueDiagnostic,
)
from data_analysis_engine.engines.validation.multicollinearity import (
    MulticollinearityDiagnostic,
)
from data_analysis_engine.engines.validation.non_finite import (
    NonFiniteValueDiagnostic,
)
from data_analysis_engine.engines.validation.outliers import (
    OutlierDiagnostic,
)
from data_analysis_engine.engines.validation.temporal_quality import (
    TemporalQualityDiagnostic,
)
from data_analysis_engine.engines.validation.type_consistency import (
    TypeConsistencyDiagnostic,
)
from data_analysis_engine.engines.validation.variance import (
    VarianceDiagnostic,
)


_RISK_ORDER = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}


def _safe_execute(
    callback: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return {
            "status": "success",
            "result": callback(),
        }
    except InvalidAnalysisInputError as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
        }


def _diagnostic_risk(
    diagnostic: dict[str, Any],
) -> str:
    risk = diagnostic.get(
        "risk_level"
    )

    if risk is None:
        risk = diagnostic.get(
            "severity"
        )

    if risk not in _RISK_ORDER:
        return "none"

    return str(risk)


def _is_numeric_series(
    series: pd.Series,
) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(
            series.dtype
        )
        and not pd.api.types.is_bool_dtype(
            series.dtype
        )
    )


def _is_categorical_series(
    series: pd.Series,
) -> bool:
    return bool(
        pd.api.types.is_object_dtype(
            series.dtype
        )
        or isinstance(
            series.dtype,
            pd.CategoricalDtype,
        )
        or pd.api.types.is_string_dtype(
            series.dtype
        )
        or pd.api.types.is_bool_dtype(
            series.dtype
        )
    )


def _is_temporal_series(
    series: pd.Series,
) -> bool:
    return bool(
        pd.api.types.is_datetime64_any_dtype(
            series.dtype
        )
    )


class DataQualityReport(
    AnalysisMethod
):
    name = "data_quality_report"
    category = "validation"
    version = "1.0"

    description = (
        "Aggregate native data-quality diagnostics into "
        "a dataset-level report covering completeness, "
        "duplicates, consistency, cardinality, numerical "
        "quality, temporal quality, and multicollinearity."
    )

    supported_roles = (
        "time",
        "measure",
        "dimension",
        "text",
    )

    minimum_compatible_columns = 1

    selection_mode = "joint"
    analysis_unit = "dataset"

    role_requirements = ()

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        data = kwargs.get("data")

        if not isinstance(
            data,
            dict,
        ):
            raise InvalidAnalysisInputError(
                "'data' must be a dictionary "
                "of column names to values."
            )

        if not data:
            raise InvalidAnalysisInputError(
                "Data-quality reports require "
                "at least one column."
            )

        try:
            frame = pd.DataFrame(
                data
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "All columns must form a rectangular "
                "dataframe."
            ) from exc

        row_count = int(
            len(frame)
        )

        column_count = int(
            frame.shape[1]
        )

        if row_count == 0:
            raise InvalidAnalysisInputError(
                "Data-quality reports require "
                "at least one row."
            )

        columns = [
            str(column)
            for column
            in frame.columns
        ]

        column_reports: dict[
            str,
            dict[str, Any],
        ] = {}

        risk_counts = {
            "none": 0,
            "low": 0,
            "moderate": 0,
            "high": 0,
            "critical": 0,
        }

        issue_count = 0
        executed_check_count = 0
        skipped_check_count = 0

        def register_check(
            container: dict[str, Any],
            name: str,
            check: dict[str, Any],
        ) -> None:
            nonlocal issue_count
            nonlocal executed_check_count
            nonlocal skipped_check_count

            container[name] = check

            if check[
                "status"
            ] != "success":
                skipped_check_count += 1
                return

            executed_check_count += 1

            risk = _diagnostic_risk(
                check["result"]
            )

            risk_counts[risk] += 1

            if risk != "none":
                issue_count += 1

        for original_column in frame.columns:
            column = str(
                original_column
            )

            series = frame[
                original_column
            ]

            report: dict[
                str,
                Any,
            ] = {
                "dtype": str(
                    series.dtype
                ),
                "checks": {},
            }

            checks = report[
                "checks"
            ]

            register_check(
                checks,
                "missingness",
                _safe_execute(
                    lambda series=series: (
                        MissingValueDiagnostic().execute(
                            values=series
                        )
                    )
                ),
            )

            register_check(
                checks,
                "constantness",
                _safe_execute(
                    lambda series=series: (
                        ConstantColumnDiagnostic().execute(
                            values=series
                        )
                    )
                ),
            )

            register_check(
                checks,
                "type_consistency",
                _safe_execute(
                    lambda series=series: (
                        TypeConsistencyDiagnostic().execute(
                            values=series
                        )
                    )
                ),
            )

            if _is_categorical_series(
                series
            ):
                register_check(
                    checks,
                    "cardinality",
                    _safe_execute(
                        lambda series=series: (
                            CardinalityDiagnostic().execute(
                                values=series
                            )
                        )
                    ),
                )

            if _is_numeric_series(
                series
            ):
                register_check(
                    checks,
                    "non_finite",
                    _safe_execute(
                        lambda series=series: (
                            NonFiniteValueDiagnostic().execute(
                                values=series
                            )
                        )
                    ),
                )

                register_check(
                    checks,
                    "outliers",
                    _safe_execute(
                        lambda series=series: (
                            OutlierDiagnostic().execute(
                                values=series
                            )
                        )
                    ),
                )

                register_check(
                    checks,
                    "variance",
                    _safe_execute(
                        lambda series=series: (
                            VarianceDiagnostic().execute(
                                values=series
                            )
                        )
                    ),
                )

            if _is_temporal_series(
                series
            ):
                register_check(
                    checks,
                    "temporal_quality",
                    _safe_execute(
                        lambda series=series: (
                            TemporalQualityDiagnostic().execute(
                                values=series
                            )
                        )
                    ),
                )

            column_reports[
                column
            ] = report

        dataset_checks: dict[
            str,
            Any,
        ] = {}

        duplicate_data = {
            str(column): frame[
                column
            ].tolist()
            for column
            in frame.columns
        }

        register_check(
            dataset_checks,
            "duplicate_rows",
            _safe_execute(
                lambda: (
                    DuplicateRowDiagnostic().execute(
                        data=duplicate_data
                    )
                )
            ),
        )

        numeric_columns = [
            column
            for column
            in frame.columns
            if _is_numeric_series(
                frame[column]
            )
        ]

        if len(
            numeric_columns
        ) >= 2:
            numeric_data = {
                str(column): frame[
                    column
                ].tolist()
                for column
                in numeric_columns
            }

            register_check(
                dataset_checks,
                "multicollinearity",
                _safe_execute(
                    lambda: (
                        MulticollinearityDiagnostic().execute(
                            data=numeric_data
                        )
                    )
                ),
            )

        maximum_risk_score = max(
            (
                _RISK_ORDER[
                    risk
                ]
                for risk, count
                in risk_counts.items()
                if count > 0
            ),
            default=0,
        )

        overall_risk_level = next(
            risk
            for risk, score
            in _RISK_ORDER.items()
            if score
            == maximum_risk_score
        )

        if overall_risk_level == "critical":
            quality_status = (
                "critical_issues"
            )
        elif overall_risk_level == "high":
            quality_status = (
                "major_issues"
            )
        elif overall_risk_level == "moderate":
            quality_status = (
                "review_required"
            )
        elif overall_risk_level == "low":
            quality_status = (
                "minor_issues"
            )
        else:
            quality_status = "clean"

        return {
            "diagnostic_type": (
                "data_quality_report"
            ),
            "row_count": row_count,
            "column_count": column_count,
            "columns": columns,
            "column_reports": (
                column_reports
            ),
            "dataset_checks": (
                dataset_checks
            ),
            "numeric_column_count": int(
                len(
                    numeric_columns
                )
            ),
            "executed_check_count": (
                executed_check_count
            ),
            "skipped_check_count": (
                skipped_check_count
            ),
            "issue_count": (
                issue_count
            ),
            "risk_counts": (
                risk_counts
            ),
            "overall_risk_level": (
                overall_risk_level
            ),
            "quality_status": (
                quality_status
            ),
            "report_scope": {
                "missing_values": True,
                "constant_columns": True,
                "duplicate_rows": True,
                "non_finite_values": True,
                "outliers": True,
                "cardinality": True,
                "type_consistency": True,
                "temporal_quality": True,
                "variance": True,
                "multicollinearity": True,
            },
        }
