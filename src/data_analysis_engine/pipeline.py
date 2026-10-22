from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data_analysis_engine.analysis_report import (
    DatasetAnalysisReport,
)
from data_analysis_engine.datasets import (
    Dataset,
    DatasetLoader,
)
from data_analysis_engine.engine import (
    DataAnalysisEngine,
)
from data_analysis_engine.orchestration import (
    AnalysisExecutor,
    AnalysisPlanner,
    AnalyticalRouter,
)
from data_analysis_engine.profiling import (
    DataProfiler,
)


class DataAnalysisPipeline:
    """
    High-level framework-independent analytical pipeline.
    """

    def __init__(
        self,
        engine: DataAnalysisEngine | None = None,
        profiler: DataProfiler | None = None,
        router: AnalyticalRouter | None = None,
        planner: AnalysisPlanner | None = None,
        executor: AnalysisExecutor | None = None,
    ) -> None:
        self.engine = engine or DataAnalysisEngine()
        self.profiler = profiler or DataProfiler()
        self.router = router or AnalyticalRouter()
        self.planner = planner or AnalysisPlanner()
        self.executor = (
            executor
            or AnalysisExecutor(self.engine)
        )

    def analyze_dataset(
        self,
        dataset: Dataset,
        *,
        methods: Iterable[str] | None = None,
        parameters_by_method: dict[
            str,
            dict[str, Any],
        ]
        | None = None,
    ) -> DatasetAnalysisReport:
        profile = self.profiler.profile(
            dataset
        )

        routing = self.router.route(
            profile
        )

        plan = self.planner.plan(
            routing,
            methods=methods,
        )

        execution = self.executor.execute(
            dataset,
            plan,
            parameters_by_method=parameters_by_method,
        )

        return DatasetAnalysisReport(
            dataset_name=dataset.name,
            dataset_source=dataset.source,
            dataset_fingerprint=dataset.fingerprint(),
            profile=profile,
            routing=routing,
            plan=plan,
            execution=execution,
        )

    def analyze_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        name: str | None = None,
        source: str | None = None,
        methods: Iterable[str] | None = None,
        parameters_by_method: dict[
            str,
            dict[str, Any],
        ]
        | None = None,
    ) -> DatasetAnalysisReport:
        dataset = DatasetLoader.from_dataframe(
            dataframe,
            name=name,
            source=source,
        )

        return self.analyze_dataset(
            dataset,
            methods=methods,
            parameters_by_method=parameters_by_method,
        )

    def analyze_csv(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        methods: Iterable[str] | None = None,
        parameters_by_method: dict[
            str,
            dict[str, Any],
        ]
        | None = None,
        **read_csv_options: Any,
    ) -> DatasetAnalysisReport:
        dataset = DatasetLoader.from_csv(
            path,
            name=name,
            **read_csv_options,
        )

        return self.analyze_dataset(
            dataset,
            methods=methods,
            parameters_by_method=parameters_by_method,
        )
