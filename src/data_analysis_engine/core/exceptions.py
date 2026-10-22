class DataAnalysisError(Exception):
    """Base exception for the engine."""


class MethodNotFoundError(DataAnalysisError):
    pass


class MethodAlreadyRegisteredError(DataAnalysisError):
    pass


class InvalidAnalysisInputError(DataAnalysisError):
    pass


class PlanDatasetMismatchError(DataAnalysisError):
    """Raised when an analysis plan belongs to another dataset."""

    pass
