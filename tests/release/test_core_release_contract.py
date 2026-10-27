import json
from pathlib import Path

from data_analysis_engine import DataAnalysisEngine


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "docs"
    / "native_methods_manifest.json"
)


def registry():
    return DataAnalysisEngine().available_methods()


def manifest():
    return json.loads(
        MANIFEST_PATH.read_text()
    )


def test_core_v1_manifest_has_exactly_95_capabilities():
    frozen = manifest()

    assert frozen["native_method_count"] == 95
    assert len(frozen["methods"]) == 95


def test_core_v1_manifest_method_names_are_unique():
    names = [
        method["name"]
        for method in manifest()["methods"]
    ]

    assert len(names) == 95
    assert len(set(names)) == 95


def test_core_v1_manifest_remains_present_in_live_registry():
    live_names = {
        method["name"]
        for method in registry()
    }

    manifest_names = [
        method["name"]
        for method
        in manifest()["methods"]
    ]

    assert manifest_names == sorted(manifest_names)
    assert set(manifest_names).issubset(live_names)


def test_core_v1_contains_priority_categories():
    categories = {
        method["category"]
        for method in registry()
    }

    assert {
        "validation",
        "preprocessing",
        "statistics",
        "regression",
    }.issubset(categories)


def test_core_v1_contains_priority_capabilities():
    names = {
        method["name"]
        for method in registry()
    }

    required = {
        "data_quality_report",
        "missing_value_imputation",
        "train_test_split",
        "distribution_shape_statistics",
        "bootstrap_mean_confidence_interval",
        "welch_mean_difference_t_test",
        "chi_square_independence_test",
        "ordinary_least_squares",
        "ordinary_least_squares_inference",
        "breusch_pagan_heteroskedasticity_test",
        "newey_west_ols_inference",
        "nested_ols_f_test",
        "jarque_bera_residual_normality_test",
    }

    assert required.issubset(names)


def test_registry_bootstrap_is_idempotent():
    first = [
        method["name"]
        for method
        in DataAnalysisEngine().available_methods()
    ]

    second = [
        method["name"]
        for method
        in DataAnalysisEngine().available_methods()
    ]

    assert first == second
    assert len(first) >= manifest()["native_method_count"]
