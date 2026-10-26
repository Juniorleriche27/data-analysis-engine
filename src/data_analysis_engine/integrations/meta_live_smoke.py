from __future__ import annotations

import json
import os
import sys

import pandas as pd

from data_analysis_engine.integrations.meta_model_api import MetaModelAPIClient
from data_analysis_engine.orchestration.meta_workflow import MetaAnalysisWorkflow


class MetaLiveSmokeError(RuntimeError):
    pass


def require_model_api_key() -> str:
    api_key = os.environ.get("MODEL_API_KEY")

    if not isinstance(api_key, str) or not api_key.strip():
        raise MetaLiveSmokeError("MODEL_API_KEY is not configured.")

    return api_key.strip()


def build_demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sales": [
                10.0,
                20.0,
                30.0,
                40.0,
                50.0,
            ]
        }
    )


def run_live_smoke() -> dict:
    require_model_api_key()

    workflow = MetaAnalysisWorkflow(
        MetaModelAPIClient()
    )

    return workflow.run(
        dataframe=build_demo_dataframe(),
        user_request=(
            "Calculate the median of the sales column. "
            "Use the native advanced_quantiles method, "
            "bind values to the sales column, and explain "
            "the deterministic result."
        ),
    )


def main() -> int:
    try:
        result = run_live_smoke()
    except Exception as exc:
        print(f"LIVE SMOKE FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
