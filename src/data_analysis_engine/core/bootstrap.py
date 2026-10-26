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
from data_analysis_engine.engines.timeseries.anomalies import (
    RollingMADAnomaly,
    RollingZScoreAnomaly,
)
from data_analysis_engine.engines.timeseries.ar_selection import (
    AutoregressiveOrderSelection,
)
from data_analysis_engine.engines.timeseries.autoregression import (
    AutoregressiveForecast,
)
from data_analysis_engine.engines.timeseries.basic import (
    Autocorrelation,
    AutocorrelationFunction,
    LaggedSeries,
    LogReturns,
    RollingStatistics,
    SimpleReturns,
)
from data_analysis_engine.engines.timeseries.change_points import (
    MeanShiftChangePoint,
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
from data_analysis_engine.engines.timeseries.exponential_smoothing import (
    SimpleExponentialSmoothingForecast,
)
from data_analysis_engine.engines.timeseries.forecast_evaluation import (
    RollingOriginForecastBacktest,
)
from data_analysis_engine.engines.timeseries.forecast_intervals import (
    EmpiricalForecastInterval,
)
from data_analysis_engine.engines.timeseries.conformal_intervals import (
    ConformalForecastInterval,
)
from data_analysis_engine.engines.timeseries.conformal_calibration import (
    ConformalIntervalCalibration,
)
from data_analysis_engine.engines.timeseries.model_selection import (
    ForecastModelSelection,
)
from data_analysis_engine.engines.timeseries.automatic_forecast import (
    AutomaticForecast,
)
from data_analysis_engine.engines.timeseries.automatic_forecast_intervals import (
    AutomaticForecastWithIntervals,
)
from data_analysis_engine.engines.timeseries.automatic_forecast_reliability import (
    AutomaticForecastReliability,
)
from data_analysis_engine.engines.timeseries.forecast_ensemble import (
    WeightedForecastEnsemble,
)
from data_analysis_engine.engines.timeseries.ensemble_backtest import (
    AdaptiveForecastEnsembleBacktest,
)
from data_analysis_engine.engines.timeseries.strategy_selection import (
    ForecastStrategySelection,
)
from data_analysis_engine.engines.timeseries.automatic_strategy_forecast import (
    AutomaticStrategyForecast,
)
from data_analysis_engine.engines.timeseries.automatic_strategy_intervals import (
    AutomaticStrategyForecastWithIntervals,
)
from data_analysis_engine.engines.timeseries.automatic_strategy_reliability import (
    AutomaticStrategyForecastReliability,
)
from data_analysis_engine.engines.validation.missingness import (
    MissingValueDiagnostic,
)
from data_analysis_engine.engines.validation.constantness import (
    ConstantColumnDiagnostic,
)
from data_analysis_engine.engines.validation.constant_columns import (
    ConstantColumnDiagnostic,
)
from data_analysis_engine.engines.validation.duplicate_rows import (
    DuplicateRowDiagnostic,
)
from data_analysis_engine.engines.validation.non_finite import (
    NonFiniteValueDiagnostic,
)
from data_analysis_engine.engines.timeseries.interval_calibration import (
    ForecastIntervalCalibration,
)
from data_analysis_engine.engines.timeseries.forecasting import (
    DriftForecast,
    NaiveForecast,
    SeasonalNaiveForecast,
)
from data_analysis_engine.engines.timeseries.holt import (
    HoltLinearTrendForecast,
)
from data_analysis_engine.engines.timeseries.holt_winters import (
    HoltWintersAdditiveForecast,
)
from data_analysis_engine.engines.timeseries.holt_winters_selection import (
    HoltWintersAdditiveParameterSelection,
)
from data_analysis_engine.engines.timeseries.holt_selection import (
    HoltParameterSelection,
)
from data_analysis_engine.engines.timeseries.smoothing_selection import (
    SimpleExponentialSmoothingAlphaSelection,
)
from data_analysis_engine.engines.timeseries.stability import (
    TimeSeriesStabilityDiagnostic,
)
from data_analysis_engine.engines.timeseries.temporal_structure import (
    TemporalStructureDiagnostic,
)
from data_analysis_engine.engines.timeseries.trend import (
    LinearDetrend,
    LinearTrend,
)
from data_analysis_engine.engines.timeseries.variance_change import (
    VarianceShiftChangePoint,
)
from data_analysis_engine.engines.timeseries.volatility import (
    EWMAVolatility,
    RollingVolatility,
)



from data_analysis_engine.engines.validation.outliers import (
    OutlierDiagnostic,
)


from data_analysis_engine.engines.validation.cardinality import (
    CardinalityDiagnostic,
)


from data_analysis_engine.engines.validation.type_consistency import (
    TypeConsistencyDiagnostic,
)


from data_analysis_engine.engines.validation.temporal_quality import (
    TemporalQualityDiagnostic,
)


from data_analysis_engine.engines.validation.variance import (
    VarianceDiagnostic,
)


from data_analysis_engine.engines.validation.multicollinearity import (
    MulticollinearityDiagnostic,
)


from data_analysis_engine.engines.validation.quality_report import (
    DataQualityReport,
)


from data_analysis_engine.engines.preprocessing.imputation import (
    MissingValueImputation,
)


from data_analysis_engine.engines.preprocessing.scaling import (
    FeatureScaling,
)


from data_analysis_engine.engines.preprocessing.transformations import (
    NumericalTransformation,
)


from data_analysis_engine.engines.preprocessing.winsorization import (
    Winsorization,
)


from data_analysis_engine.engines.preprocessing.categorical_encoding import (
    CategoricalEncoding,
)


from data_analysis_engine.engines.preprocessing.splitting import (
    TrainTestSplit,
)


from data_analysis_engine.engines.inference.distribution_shape import (
    DistributionShapeStatistics,
)


from data_analysis_engine.engines.inference.quantiles import (
    AdvancedQuantiles,
)


from data_analysis_engine.engines.inference.confidence_intervals import (
    MeanConfidenceInterval,
)


from data_analysis_engine.engines.inference.bootstrap import (
    BootstrapMeanConfidenceInterval,
)


from data_analysis_engine.engines.inference.permutation import (
    PermutationMeanDifferenceTest,
)


from data_analysis_engine.engines.inference.parametric_tests import (
    OneSampleMeanTTest,
    WelchMeanDifferenceTTest,
    PairedMeanDifferenceTTest,
)


from data_analysis_engine.engines.inference.proportion_tests import (
    OneSampleProportionZTest,
    TwoSampleProportionZTest,
)


from data_analysis_engine.engines.inference.chi_square import (
    ChiSquareIndependenceTest,
)


from data_analysis_engine.engines.regression.ols import (
    OrdinaryLeastSquares,
)


from data_analysis_engine.engines.regression.ols_inference import (
    OrdinaryLeastSquaresInference,
)


from data_analysis_engine.engines.regression.diagnostics import (
    BreuschPaganHeteroskedasticityTest,
    OLSResidualAutocorrelationDiagnostic,
    OLSInfluenceDiagnostic,
)


from data_analysis_engine.engines.regression.econometrics import (
    NeweyWestOLSInference,
    NestedOLSFTest,
    JarqueBeraResidualNormalityTest,
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
    TimeSeriesStabilityDiagnostic,
    RollingVolatility,
    EWMAVolatility,
    MeanShiftChangePoint,
    VarianceShiftChangePoint,
    RollingZScoreAnomaly,
    RollingMADAnomaly,
    TemporalStructureDiagnostic,
    NaiveForecast,
    DriftForecast,
    SeasonalNaiveForecast,
    RollingOriginForecastBacktest,
    AutoregressiveForecast,
    AutoregressiveOrderSelection,
    SimpleExponentialSmoothingForecast,
    SimpleExponentialSmoothingAlphaSelection,
    HoltLinearTrendForecast,
    HoltParameterSelection,
    HoltWintersAdditiveForecast,
    HoltWintersAdditiveParameterSelection,
    EmpiricalForecastInterval,
    ForecastIntervalCalibration,
    ConformalForecastInterval,
    ConformalIntervalCalibration,
    ForecastModelSelection,
    AutomaticForecast,
    AutomaticForecastWithIntervals,
    AutomaticForecastReliability,
    WeightedForecastEnsemble,
    AdaptiveForecastEnsembleBacktest,
    ForecastStrategySelection,
    AutomaticStrategyForecast,
    AutomaticStrategyForecastWithIntervals,
    AutomaticStrategyForecastReliability,
    MissingValueDiagnostic,
    ConstantColumnDiagnostic,
    DuplicateRowDiagnostic,
    NonFiniteValueDiagnostic,
    OutlierDiagnostic,
    CardinalityDiagnostic,
    TypeConsistencyDiagnostic,
    TemporalQualityDiagnostic,
    VarianceDiagnostic,
    MulticollinearityDiagnostic,
    DataQualityReport,
    MissingValueImputation,
    FeatureScaling,
    NumericalTransformation,
    Winsorization,
    CategoricalEncoding,
    TrainTestSplit,
    DistributionShapeStatistics,
    AdvancedQuantiles,
    MeanConfidenceInterval,
    BootstrapMeanConfidenceInterval,
    PermutationMeanDifferenceTest,
    OneSampleMeanTTest,
    WelchMeanDifferenceTTest,
    PairedMeanDifferenceTTest,
    OneSampleProportionZTest,
    TwoSampleProportionZTest,
    ChiSquareIndependenceTest,
    OrdinaryLeastSquares,
    OrdinaryLeastSquaresInference,
    BreuschPaganHeteroskedasticityTest,
    OLSResidualAutocorrelationDiagnostic,
    OLSInfluenceDiagnostic,
    NeweyWestOLSInference,
    NestedOLSFTest,
    JarqueBeraResidualNormalityTest,
)


def register_builtin_methods() -> None:
    for method_class in BUILTIN_METHODS:
        method = method_class()

        if not registry.exists(method.name):
            registry.register(method)
