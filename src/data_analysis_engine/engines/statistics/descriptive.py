from statistics import mean, median, pvariance, pstdev
from typing import Any

from data_analysis_engine.core.contracts import AnalysisMethod
from data_analysis_engine.core.exceptions import InvalidAnalysisInputError


class DescriptiveStatistics(AnalysisMethod):
    name = "descriptive_statistics"
    category = "statistics"
    version = "1.0"

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        values = kwargs.get("values")

        if values is None:
            raise InvalidAnalysisInputError("'values' is required.")

        if not isinstance(values, (list, tuple)):
            raise InvalidAnalysisInputError(
                "'values' must be a list or tuple."
            )

        if not values:
            raise InvalidAnalysisInputError(
                "'values' cannot be empty."
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in values
        ):
            raise InvalidAnalysisInputError(
                "'values' must contain only numerical values."
            )

        return {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "variance_population": pvariance(values),
            "std_population": pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
