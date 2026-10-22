from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProfilerConfig:
    """
    Configuration for semantic type inference and profiling.
    """

    categorical_ratio_threshold: float = 0.20
    small_dataset_max_rows: int = 20
    small_dataset_categorical_ratio_threshold: float = 0.50

    high_missing_percentage_warning: float = 30.0
