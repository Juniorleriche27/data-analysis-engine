"""
Backward-compatible import path for the constant-column diagnostic.

The canonical implementation lives in:
data_analysis_engine.engines.validation.constantness
"""

from data_analysis_engine.engines.validation.constantness import (
    ConstantColumnDiagnostic,
)

__all__ = [
    "ConstantColumnDiagnostic",
]
