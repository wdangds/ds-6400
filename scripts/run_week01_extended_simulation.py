#!/usr/bin/env python
"""Run the Week 1 extended simulation outside the Quarto kernel.

The script writes progress to simulation-progress.log and checkpoints each
completed (n, dgp, model) setting to simulation-cache/.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd


SAMPLE_SIZES = [20, 30, 50, 100, 200]
DGPS = ["efron", "logistic", "nonlinear"]
MODELS = ["logistic", "lda", "1nn", "rf"]

DGP_SEED_OFFSET = {
    "efron": 101,
    "logistic": 211,
    "nonlinear": 307,
}

MODEL_SEED_OFFSET = {
    "logistic": 11,
    "lda": 23,
    "1nn": 37,
    "rf": 41,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ds6400.cache import extended_results_path, setting_cache_path  # noqa: E402
from ds6400.progress import (  # noqa: E402
    prepare_simulation_cache,
    reset_progress_log,
    suppress_sklearn_warnings,
    write_progress,
)
from ds6400.simulation import run_general_setting, should_compute_loo  # noqa: E402


PROGRESS_LOG = PROJECT_ROOT / "simulation-progress.log"
SIMULATION_CACHE_DIR = PROJECT_ROOT / "simulation-cache"
LOG_START = perf_counter()
log_progress = lambda message: write_progress(message, PROGRESS_LOG, LOG_START)


def load_or_run_setting(n: int, dgp: str, model_name: str, args: argparse.Namespace) -> pd.DataFrame:
    compute_loo = should_compute_loo(model_name, args.compute_loo, args.compute_loo_rf)
    cache_path = setting_cache_path(
        cache_dir=SIMULATION_CACHE_DIR,
        n=n,
        dgp=dgp,
        model_name=model_name,
        replications=args.replications,
        bootstrap_b=args.bootstrap_b,
        compute_loo=compute_loo,
        test_n=args.test_n,
        rf_n_estimators=args.rf_trees,
    )

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) >= args.replications:
            log_progress(f"Using cached setting: n={n}, dgp={dgp}, model={model_name}")
            return cached
        log_progress(f"Ignoring incomplete cache for n={n}, dgp={dgp}, model={model_name}: {len(cached)} rows")

    seed = args.seed + n + DGP_SEED_OFFSET[dgp] + MODEL_SEED_OFFSET[model_name]
    df = run_general_setting(
        n=n,
        dgp=dgp,
        model_name=model_name,
        R=args.replications,
        bootstrap_B=args.bootstrap_b,
        compute_loo=compute_loo,
        test_n=args.test_n,
        seed=seed,
        progress_every=max(1, args.replications // 20),
        progress_callback=log_progress,
        rf_n_estimators=args.rf_trees,
    )
    df.to_csv(cache_path, index=False)
    gc.collect()
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, required=True)
    parser.add_argument("--bootstrap-b", type=int, required=True)
    parser.add_argument("--test-n", type=int, required=True)
    parser.add_argument("--rf-trees", type=int, required=True)
    parser.add_argument("--seed", type=int, default=6400)
    parser.add_argument("--compute-loo", action="store_true")
    parser.add_argument("--compute-loo-rf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suppress_sklearn_warnings()
    reset_progress_log(PROGRESS_LOG)
    prepare_simulation_cache(SIMULATION_CACHE_DIR)

    output_path = extended_results_path(
        cache_dir=SIMULATION_CACHE_DIR,
        replications=args.replications,
        bootstrap_b=args.bootstrap_b,
        test_n=args.test_n,
        rf_n_estimators=args.rf_trees,
        compute_loo_for_random_forest=args.compute_loo_rf,
    )
    if output_path.exists():
        cached = pd.read_csv(output_path)
        expected_rows = len(SAMPLE_SIZES) * len(DGPS) * len(MODELS) * args.replications
        if len(cached) >= expected_rows:
            log_progress(f"Using cached extended simulation results: {output_path.name}")
            return

    total_settings = len(SAMPLE_SIZES) * len(DGPS) * len(MODELS)
    all_model_results = []

    log_progress(
        f"Starting extended simulation: {total_settings} settings, "
        f"R={args.replications}, bootstrap_B={args.bootstrap_b}, "
        f"LOO={args.compute_loo}, LOO_RF={args.compute_loo_rf}, "
        f"test_n={args.test_n}, rf_trees={args.rf_trees}"
    )

    setting_number = 0
    for n in SAMPLE_SIZES:
        for dgp in DGPS:
            for model_name in MODELS:
                setting_number += 1
                compute_loo = should_compute_loo(model_name, args.compute_loo, args.compute_loo_rf)
                start = perf_counter()
                log_progress(
                    f"Setting {setting_number}/{total_settings}: "
                    f"n={n}, dgp={dgp}, model={model_name}, LOO={compute_loo}"
                )
                df = load_or_run_setting(n=n, dgp=dgp, model_name=model_name, args=args)
                all_model_results.append(df)
                log_progress(
                    f"Finished n={n}, dgp={dgp}, model={model_name}: "
                    f"{len(df)} rows in {perf_counter() - start:.1f}s"
                )

    model_results = pd.concat(all_model_results, ignore_index=True)
    model_results.to_csv(output_path, index=False)
    log_progress(f"Extended simulation complete: {len(model_results)} rows")
    log_progress(f"Saved combined results: {output_path}")


if __name__ == "__main__":
    main()
