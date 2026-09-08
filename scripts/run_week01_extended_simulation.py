#!/usr/bin/env python
"""Run the Week 1 extended simulation outside the Quarto kernel.

The script writes progress to simulation-progress.log and checkpoints each
completed (n, dgp, model) setting to simulation-cache/.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import (
    ConvergenceWarning,
    DataConversionWarning,
    FitFailedWarning,
    UndefinedMetricWarning,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


for warning_category in [
    ConvergenceWarning,
    DataConversionWarning,
    FitFailedWarning,
    UndefinedMetricWarning,
]:
    warnings.filterwarnings("ignore", category=warning_category)

warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\..*")


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
PROGRESS_LOG = PROJECT_ROOT / "simulation-progress.log"
SIMULATION_CACHE_DIR = PROJECT_ROOT / "simulation-cache"
LOG_START = perf_counter()


def log_progress(message: str) -> None:
    elapsed = perf_counter() - LOG_START
    line = f"[{elapsed:8.1f}s] {message}"
    with PROGRESS_LOG.open("a", encoding="utf-8") as f:
        print(line, file=f, flush=True)


def generate_efron_dgp(n: int, rng: np.random.Generator):
    y = rng.binomial(n=1, p=0.5, size=n)
    x1 = rng.normal(loc=y - 0.5, scale=1, size=n)
    x2 = rng.normal(loc=0, scale=1, size=n)
    x = np.column_stack((x1, x2))
    pi_true = expit(x1)
    return x, y, pi_true


def generate_logistic_dgp(n: int, rng: np.random.Generator):
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = -0.2 + x1 - 0.7 * x2
    pi_true = expit(eta)
    y = rng.binomial(1, pi_true)
    x = np.column_stack([x1, x2])
    return x, y, pi_true


def generate_nonlinear_dgp(n: int, rng: np.random.Generator):
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = -0.2 + x1 - 0.7 * x2 + 1.1 * (x1**2 - 1)
    pi_true = expit(eta)
    y = rng.binomial(1, pi_true)
    x = np.column_stack([x1, x2])
    return x, y, pi_true


def generate_data(n: int, dgp: str, rng: np.random.Generator):
    if dgp == "efron":
        return generate_efron_dgp(n, rng)
    if dgp == "logistic":
        return generate_logistic_dgp(n, rng)
    if dgp == "nonlinear":
        return generate_nonlinear_dgp(n, rng)
    raise ValueError(f"Unknown DGP: {dgp}")


def make_model(model_name: str, rf_trees: int, seed: int | None = None):
    if model_name == "logistic":
        return LogisticRegression(penalty=None, solver="lbfgs", random_state=seed)
    if model_name == "lda":
        return LinearDiscriminantAnalysis()
    if model_name == "1nn":
        return KNeighborsClassifier(n_neighbors=1)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=None,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def apparent_error(model, x, y) -> float:
    y_pred = model.predict(x)
    return float(np.mean(y_pred != y))


def true_fixed_x_error(model, x, pi_true) -> float:
    y_hat = model.predict(x)
    error_probability = np.where(y_hat == 1, 1 - pi_true, pi_true)
    return float(np.mean(error_probability))


def efron_optimism(model, x, cutoff: float = 0.5) -> float:
    n = len(x)
    t = np.column_stack((np.ones(n), x))
    beta_hat = np.concatenate([model.intercept_, model.coef_.ravel()])
    pi_hat = model.predict_proba(x)[:, 1]
    w = pi_hat * (1 - pi_hat)
    xi_hat = t.T @ (w[:, None] * t)
    xi_inv = np.linalg.inv(xi_hat)
    d_hat = np.einsum("ij,jk,ik->i", t, xi_inv, t)
    d_hat = np.maximum(d_hat, 1e-10)
    logit_cutoff = np.log(cutoff / (1 - cutoff))
    c_hat = logit_cutoff - t @ beta_hat
    return float((2 / n) * np.sum(w * norm.pdf(c_hat / np.sqrt(d_hat)) * np.sqrt(d_hat)))


def loo_cv_error(model, x, y) -> float:
    loo = LeaveOneOut()
    scores = cross_val_score(model, x, y, cv=loo, scoring="accuracy")
    return float(1 - scores.mean())


def kfold_cv_error(model, x, y, k: int = 10, seed: int = 0) -> float:
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or class_counts.min() < k:
        return np.nan
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(model), x, y, cv=cv, scoring="accuracy")
    return float(1 - scores.mean())


def bootstrap_corrected_error(model, x, y, rf_trees: int, b_count: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(y)
    original_model = clone(model)
    original_model.fit(x, y)
    apparent = apparent_error(original_model, x, y)
    optimism_values = []

    for _ in range(b_count):
        idx = rng.integers(0, n, size=n)
        x_boot = x[idx]
        y_boot = y[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        boot_model = clone(model)
        try:
            boot_model.fit(x_boot, y_boot)
        except Exception:
            continue

        err_boot = np.mean(boot_model.predict(x_boot) != y_boot)
        err_original = np.mean(boot_model.predict(x) != y)
        optimism_values.append(err_original - err_boot)

    if len(optimism_values) == 0:
        return np.nan
    return float(apparent + np.mean(optimism_values))


def independent_test_error(model, dgp: str, rng: np.random.Generator, n_test: int) -> float:
    x_test, y_test, _ = generate_data(n_test, dgp, rng)
    y_hat = model.predict(x_test)
    return float(np.mean(y_hat != y_test))


def one_general_replication(
    n: int,
    dgp: str,
    model_name: str,
    rng: np.random.Generator,
    rf_trees: int,
    bootstrap_b: int,
    compute_loo: bool,
    test_n: int,
):
    x, y, pi_true = generate_data(n, dgp, rng)

    if len(np.unique(y)) < 2:
        return None

    rep_seed = int(rng.integers(0, 2**32 - 1))
    model = make_model(model_name, rf_trees=rf_trees, seed=rep_seed)

    try:
        model.fit(x, y)
    except Exception:
        return None

    apparent = apparent_error(model, x, y)
    true_fixed = true_fixed_x_error(model, x, pi_true)
    actual_op = true_fixed - apparent

    try:
        cv10 = kfold_cv_error(model, x, y, k=10, seed=rep_seed)
    except Exception:
        cv10 = np.nan

    if compute_loo:
        try:
            loo = loo_cv_error(model, x, y)
        except Exception:
            loo = np.nan
    else:
        loo = np.nan

    try:
        boot = bootstrap_corrected_error(model, x, y, rf_trees=rf_trees, b_count=bootstrap_b, seed=rep_seed)
    except Exception:
        boot = np.nan

    try:
        test = independent_test_error(model, dgp, rng, n_test=test_n)
    except Exception:
        test = np.nan

    if model_name == "logistic":
        try:
            efron_op = efron_optimism(model, x)
            efron_err = apparent + efron_op
        except Exception:
            efron_op = np.nan
            efron_err = np.nan
    else:
        efron_op = np.nan
        efron_err = np.nan

    return {
        "n": n,
        "dgp": dgp,
        "model": model_name,
        "true_fixed_x": true_fixed,
        "apparent_error": apparent,
        "actual_optimism": actual_op,
        "efron_optimism": efron_op,
        "efron_error": efron_err,
        "cv10_error": cv10,
        "loo_error": loo,
        "bootstrap_error": boot,
        "test_error": test,
    }


def should_compute_loo(model_name: str, compute_loo: bool, compute_loo_rf: bool) -> bool:
    if not compute_loo:
        return False
    if model_name == "rf" and not compute_loo_rf:
        return False
    return True


def setting_cache_path(
    n: int,
    dgp: str,
    model_name: str,
    r_count: int,
    bootstrap_b: int,
    compute_loo: bool,
    test_n: int,
    rf_trees: int,
) -> Path:
    loo_flag = "loo" if compute_loo else "no-loo"
    return (
        SIMULATION_CACHE_DIR
        / (
            f"extended_n-{n}_dgp-{dgp}_model-{model_name}"
            f"_R-{r_count}_B-{bootstrap_b}_{loo_flag}"
            f"_test-{test_n}_rf-{rf_trees}.csv"
        )
    )


def combined_results_path(args: argparse.Namespace) -> Path:
    loo_rf_flag = "loo-rf" if args.compute_loo_rf else "no-loo-rf"
    return (
        SIMULATION_CACHE_DIR
        / (
            f"extended_results_R-{args.replications}_B-{args.bootstrap_b}"
            f"_test-{args.test_n}_rf-{args.rf_trees}_{loo_rf_flag}.csv"
        )
    )


def run_general_setting(
    n: int,
    dgp: str,
    model_name: str,
    r_count: int,
    rf_trees: int,
    bootstrap_b: int,
    compute_loo: bool,
    test_n: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    rows = []
    progress_every = max(1, r_count // 20)

    while len(rows) < r_count:
        result = one_general_replication(
            n=n,
            dgp=dgp,
            model_name=model_name,
            rng=rng,
            rf_trees=rf_trees,
            bootstrap_b=bootstrap_b,
            compute_loo=compute_loo,
            test_n=test_n,
        )
        if result is not None:
            result["replication"] = len(rows)
            rows.append(result)
            if len(rows) % progress_every == 0:
                log_progress(f"{dgp}/{model_name}/n={n}: {len(rows)}/{r_count} replications")
        gc.collect()

    return pd.DataFrame(rows)


def load_or_run_setting(n: int, dgp: str, model_name: str, args: argparse.Namespace) -> pd.DataFrame:
    compute_loo = should_compute_loo(model_name, args.compute_loo, args.compute_loo_rf)
    cache_path = setting_cache_path(
        n=n,
        dgp=dgp,
        model_name=model_name,
        r_count=args.replications,
        bootstrap_b=args.bootstrap_b,
        compute_loo=compute_loo,
        test_n=args.test_n,
        rf_trees=args.rf_trees,
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
        r_count=args.replications,
        rf_trees=args.rf_trees,
        bootstrap_b=args.bootstrap_b,
        compute_loo=compute_loo,
        test_n=args.test_n,
        seed=seed,
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
    PROGRESS_LOG.write_text("", encoding="utf-8")
    SIMULATION_CACHE_DIR.mkdir(exist_ok=True)

    output_path = combined_results_path(args)
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
