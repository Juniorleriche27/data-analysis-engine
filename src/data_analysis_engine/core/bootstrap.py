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
    AutocorrelationFunction,
    LaggedSeries,
    LogReturns,
    RollingStatistics,
    SimpleReturns,
)
from data_analysis_engine.engines.timeseries.cross_correlation import (
    CrossCorrelation,
)
from data_analysis_engine.engines.timeseries.decomposition import (
    AdditiveSeasonalDecomposition,
    MovingAverageTrend,
)
from data_analysis_engine.engines.timeseries.differencing import (
    FirstDifference,
    SeasonalDifference,
)
from data_analysis_engine.engines.timeseries.dtw import (
    DynamicTimeWarping,
)
from data_analysis_engine.engines.timeseries.trend import (
    LinearDetrend,
    LinearTrend,
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
    LaggedSeries,
    AutocorrelationFunction,
    DynamicTimeWarping,
    CrossCorrelation,
    FirstDifference,
    SeasonalDifference,
    MovingAverageTrend,
    AdditiveSeasonalDecomposition,
    LinearTrend,
    LinearDetrend,
)


def register_builtin_methods() -> None:
    for method_class in BUILTIN_METHODS:
        method = method_class()

        if not registry.exists(method.name):
            registry.register(method)
