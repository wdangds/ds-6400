"""Cache path helpers for simulation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def sample_sizes_slug(sample_sizes: Sequence[int]) -> str:
    """Build a stable filename fragment for a sequence of sample sizes."""
    return "n-" + "-".join(str(n) for n in sample_sizes)


def extended_results_path(
    cache_dir: Path,
    sample_sizes: Sequence[int],
    replications: int,
    bootstrap_b: int,
    test_n: int,
    rf_n_estimators: int,
    compute_loo_for_random_forest: bool,
) -> Path:
    """Build the combined extended-simulation cache path."""
    loo_rf_flag = "loo-rf" if compute_loo_for_random_forest else "no-loo-rf"
    return (
        cache_dir
        / (
            f"extended_results_{sample_sizes_slug(sample_sizes)}"
            f"_R-{replications}_B-{bootstrap_b}"
            f"_test-{test_n}_rf-{rf_n_estimators}_{loo_rf_flag}.csv"
        )
    )


def setting_cache_path(
    cache_dir: Path,
    n: int,
    dgp: str,
    model_name: str,
    replications: int,
    bootstrap_b: int,
    compute_loo: bool,
    test_n: int,
    rf_n_estimators: int,
) -> Path:
    """Build the cache path for one extended simulation setting."""
    loo_flag = "loo" if compute_loo else "no-loo"
    return (
        cache_dir
        / (
            f"extended_n-{n}_dgp-{dgp}_model-{model_name}"
            f"_R-{replications}_B-{bootstrap_b}_{loo_flag}"
            f"_test-{test_n}_rf-{rf_n_estimators}.csv"
        )
    )
