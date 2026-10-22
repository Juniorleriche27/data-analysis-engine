from data_analysis_engine.core.registry import registry
from data_analysis_engine.engines.statistics.descriptive import (
    DescriptiveStatistics,
)


BUILTIN_METHODS = (
    DescriptiveStatistics,
)


def register_builtin_methods() -> None:
    for method_class in BUILTIN_METHODS:
        method = method_class()

        if not registry.exists(method.name):
            registry.register(method)
