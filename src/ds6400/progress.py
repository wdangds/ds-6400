"""Progress logging and warning controls for long simulations."""

from __future__ import annotations

import warnings
from pathlib import Path
from time import perf_counter
from typing import Callable

from sklearn.exceptions import (
    ConvergenceWarning,
    DataConversionWarning,
    FitFailedWarning,
    UndefinedMetricWarning,
)


ProgressCallback = Callable[[str], None]


def suppress_sklearn_warnings() -> None:
    """Hide sklearn warnings that are expected during Monte Carlo experiments."""
    for warning_category in [
        ConvergenceWarning,
        DataConversionWarning,
        FitFailedWarning,
        UndefinedMetricWarning,
    ]:
        warnings.filterwarnings("ignore", category=warning_category)

    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")


def reset_progress_log(progress_log: Path) -> None:
    progress_log.write_text("", encoding="utf-8")


def prepare_simulation_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(exist_ok=True)


def write_progress(message: str, progress_log: Path, start_time: float) -> None:
    elapsed = perf_counter() - start_time
    line = f"[{elapsed:8.1f}s] {message}"
    with progress_log.open("a", encoding="utf-8") as f:
        print(line, file=f, flush=True)
