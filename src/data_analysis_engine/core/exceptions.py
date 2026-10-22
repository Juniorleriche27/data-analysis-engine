class DataAnalysisError(Exception):
    """Base exception for the engine."""


class MethodNotFoundError(DataAnalysisError):
    pass


class MethodAlreadyRegisteredError(DataAnalysisError):
    pass


class InvalidAnalysisInputError(DataAnalysisError):
    pass
