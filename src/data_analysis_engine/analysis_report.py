from dataclasses import dataclass
from typing import Any

from data_analysis_engine.orchestration.execution_models import ExecutionReport
from data_analysis_engine.orchestration.models import RoutingReport
from data_analysis_engine.orchestration.planning_models import AnalysisPlan
from data_analysis_engine.profiling.models import DatasetProfile


@dataclass(slots=True)
class DatasetAnalysisReport:
    dataset_name: str | None
    dataset_source: str | None
    dataset_fingerprint: str

    profile: DatasetProfile
    routing: RoutingReport
    plan: AnalysisPlan
    execution: ExecutionReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": {
                "name": self.dataset_name,
                "source": self.dataset_source,
                "fingerprint": self.dataset_fingerprint,
            },
            "profile": self.profile.to_dict(),
            "routing": self.routing.to_dict(),
            "plan": self.plan.to_dict(),
            "execution": self.execution.to_dict(),
        }
