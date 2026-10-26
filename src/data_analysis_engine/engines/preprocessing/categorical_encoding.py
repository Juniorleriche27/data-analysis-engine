from typing import Any

import numpy as np
import pandas as pd

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import (
    InvalidAnalysisInputError,
)


_ALLOWED_STRATEGIES = {
    "ordinal",
    "one_hot",
}


def _validate_strategy(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'strategy' must be a string."
        )

    strategy = value.strip().lower()

    if strategy not in _ALLOWED_STRATEGIES:
        raise InvalidAnalysisInputError(
            "'strategy' must be one of: "
            "ordinal, one_hot."
        )

    return strategy


def _validate_missing_label(
    value: Any,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise InvalidAnalysisInputError(
            "'missing_label' must be a string."
        )

    label = value.strip()

    if not label:
        raise InvalidAnalysisInputError(
            "'missing_label' cannot be empty."
        )

    return label


def _to_python_scalar(
    value: Any,
) -> Any:
    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    return value


def _stable_categories(
    series: pd.Series,
) -> list[Any]:
    categories: list[Any] = []

    for value in series:
        if pd.isna(value):
            continue

        try:
            already_present = any(
                value == existing
                for existing
                in categories
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "Categorical encoding requires "
                "comparable scalar observations."
            ) from exc

        if not already_present:
            categories.append(
                value
            )

    return categories


class CategoricalEncoding(
    AnalysisMethod
):
    name = "categorical_encoding"
    category = "preprocessing"
    version = "1.0"

    description = (
        "Encode categorical columns using deterministic "
        "first-seen ordinal encoding or one-hot encoding "
        "with explicit missing-value representation."
    )

    supported_roles = (
        "dimension",
        "text",
    )

    minimum_compatible_columns = 1

    selection_mode = "per_column"
    analysis_unit = "column"

    role_requirements = ()

    allow_constant_columns = True
    preserve_missing_values = True

    def execute(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        values = kwargs.get(
            "values"
        )

        if values is None:
            raise InvalidAnalysisInputError(
                "'values' is required."
            )

        strategy = _validate_strategy(
            kwargs.get(
                "strategy",
                "ordinal",
            )
        )

        missing_label = _validate_missing_label(
            kwargs.get(
                "missing_label",
                "__MISSING__",
            )
        )

        try:
            series = pd.Series(
                values
            )
        except Exception as exc:
            raise InvalidAnalysisInputError(
                "'values' must be convertible "
                "to a one-dimensional series."
            ) from exc

        total_count = int(
            len(series)
        )

        if total_count == 0:
            raise InvalidAnalysisInputError(
                "Categorical encoding requires "
                "at least one observation."
            )

        missing_mask = np.asarray(
            series.isna(),
            dtype=bool,
        )

        missing_count = int(
            np.sum(
                missing_mask
            )
        )

        categories = _stable_categories(
            series
        )

        category_count = int(
            len(categories)
        )

        serialized_categories = [
            _to_python_scalar(
                category
            )
            for category
            in categories
        ]

        if strategy == "ordinal":
            mapping = {
                category: index
                for index, category
                in enumerate(
                    categories
                )
            }

            missing_code = (
                category_count
            )

            encoded_values: list[int] = []

            for value in series:
                if pd.isna(value):
                    encoded_values.append(
                        missing_code
                    )
                else:
                    try:
                        encoded_values.append(
                            int(
                                mapping[value]
                            )
                        )
                    except (
                        KeyError,
                        TypeError,
                    ) as exc:
                        raise InvalidAnalysisInputError(
                            "Categorical encoding requires "
                            "hashable scalar observations."
                        ) from exc

            return {
                "transformation_type": (
                    "categorical_encoding"
                ),
                "strategy": "ordinal",
                "total_count": total_count,
                "missing_count": missing_count,
                "category_count": category_count,
                "categories": (
                    serialized_categories
                ),
                "mapping": {
                    str(
                        _to_python_scalar(
                            category
                        )
                    ): int(index)
                    for index, category
                    in enumerate(
                        categories
                    )
                },
                "missing_label": (
                    missing_label
                ),
                "missing_code": (
                    missing_code
                ),
                "output_column_count": 1,
                "values": (
                    encoded_values
                ),
            }

        column_names = [
            str(
                _to_python_scalar(
                    category
                )
            )
            for category
            in categories
        ]

        if (
            missing_count > 0
            and missing_label
            in column_names
        ):
            raise InvalidAnalysisInputError(
                "'missing_label' conflicts with "
                "an existing category label."
            )

        encoded_columns: dict[
            str,
            list[int],
        ] = {}

        for category, column_name in zip(
            categories,
            column_names,
        ):
            encoded_columns[
                column_name
            ] = [
                int(
                    (
                        not pd.isna(value)
                    )
                    and value == category
                )
                for value
                in series
            ]

        if missing_count > 0:
            encoded_columns[
                missing_label
            ] = [
                int(is_missing)
                for is_missing
                in missing_mask
            ]

        return {
            "transformation_type": (
                "categorical_encoding"
            ),
            "strategy": "one_hot",
            "total_count": total_count,
            "missing_count": missing_count,
            "category_count": category_count,
            "categories": (
                serialized_categories
            ),
            "missing_label": (
                missing_label
            ),
            "output_column_count": int(
                len(
                    encoded_columns
                )
            ),
            "encoded_columns": (
                encoded_columns
            ),
        }
