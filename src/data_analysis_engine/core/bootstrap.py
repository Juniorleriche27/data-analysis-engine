from data_analysis_engine.core.registry import registry
from data_analysis_engine.engines.multivariate.covariance import (
    CovarianceMatrix,
    LinearCombinationVariance,
)
from data_analysis_engine.engines.multivariate.pca import (
    PrincipalComponentAnalysis,
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
from data_analysis_engine.engines.timeseries.basic import (
    Autocorrelation,
    LogReturns,
    RollingStatistics,
    SimpleReturns,
)


BUILTIN_METHODS = (
    DescriptiveStatistics,
    PearsonCorrelation,
    SpearmanCorrelation,
    CovarianceMatrix,
    LinearCombinationVariance,
    PrincipalComponentAnalysis,
    EuclideanDistance,
    ManhattanDistance,
    MinkowskiDistance,
    CosineSimilarity,
    CosineDistance,
    SimpleReturns,
    LogReturns,
    RollingStatistics,
    Autocorrelation,
)


def register_builtin_methods() -> None:
    for method_class in BUILTIN_METHODS:
        method = method_class()

        if not registry.exists(method.name):
            registry.register(method)
