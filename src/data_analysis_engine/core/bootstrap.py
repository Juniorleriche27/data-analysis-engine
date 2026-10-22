from data_analysis_engine.core.registry import registry
from data_analysis_engine.engines.multivariate.covariance import (
    CovarianceMatrix,
    LinearCombinationVariance,
)
from data_analysis_engine.engines.similarity.numeric import (
    CosineDistance,
    CosineSimilarity,
    EuclideanDistance,
    ManhattanDistance,
    MinkowskiDistance,
)
from data_analysis_engine.engines.statistics.correlation import (
    PearsonCorrelation,
    SpearmanCorrelation,
)
from data_analysis_engine.engines.statistics.descriptive import (
    DescriptiveStatistics,
)


BUILTIN_METHODS = (
    DescriptiveStatistics,
    PearsonCorrelation,
    SpearmanCorrelation,
    CovarianceMatrix,
    LinearCombinationVariance,
    EuclideanDistance,
    ManhattanDistance,
    MinkowskiDistance,
    CosineSimilarity,
    CosineDistance,
)


def register_builtin_methods() -> None:
    for method_class in BUILTIN_METHODS:
        method = method_class()

        if not registry.exists(method.name):
            registry.register(method)
