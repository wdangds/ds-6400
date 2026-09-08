"""Monte Carlo simulation loops built from reusable DGP, model, and metric helpers."""

from __future__ import annotations

import gc
from time import perf_counter

import numpy as np
import pandas as pd

from ds6400.data_generation import generate_data, generate_efron_dgp
from ds6400.evaluation import (
    apparent_error,
    bootstrap_corrected_error,
    efron_optimism,
    independent_test_error,
    kfold_cv_error,
    loo_cv_error,
    true_fixed_x_error,
)
from ds6400.models import make_model
from ds6400.progress import ProgressCallback


def one_efron_replication(
    n: int,
    rng: np.random.Generator,
    seed: int = 6400,
    rf_n_estimators: int = 100,
):
    """Run one Monte Carlo replication of the Efron experiment."""
    x, y, pi_true = generate_efron_dgp(n, rng)
    if len(np.unique(y)) < 2:
        return None

    model = make_model("logistic", seed=seed, rf_n_estimators=rf_n_estimators)
    try:
        model.fit(x, y)
    except Exception:
        return None

    apparent = apparent_error(model, x, y)
    true_error = true_fixed_x_error(model, x, pi_true)
    actual_op = true_error - apparent
    efron_op = efron_optimism(model, x)
    efron_err = apparent + efron_op
    loo_err = loo_cv_error(model, x, y)

    return {
        "n": n,
        "true_error": true_error,
        "apparent_error": apparent,
        "actual_optimism": actual_op,
        "efron_optimism": efron_op,
        "efron_error": efron_err,
        "loo_error": loo_err,
    }


def run_efron_simulation(
    R: int = 1000,
    n: int = 20,
    seed: int = 0,
    progress_every: int | None = None,
    progress_callback: ProgressCallback | None = None,
    rf_n_estimators: int = 100,
) -> pd.DataFrame:
    """Run Efron's simulation experiment."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < R:
        result = one_efron_replication(
            n=n,
            rng=rng,
            seed=seed,
            rf_n_estimators=rf_n_estimators,
        )
        if result is not None:
            rows.append(result)
            if progress_callback and progress_every and len(rows) % progress_every == 0:
                progress_callback(f"Efron replication n={n}: {len(rows)}/{R}")
    return pd.DataFrame(rows)


def run_sample_size_study(
    sample_sizes: list[int],
    R: int = 500,
    seed: int = 0,
    progress_callback: ProgressCallback | None = None,
    rf_n_estimators: int = 100,
) -> pd.DataFrame:
    """Run the Efron simulation over several sample sizes."""
    rows = []
    for n in sample_sizes:
        if progress_callback:
            progress_callback(f"Starting sample-size study n={n}: R={R}")
        start = perf_counter()
        df_n = run_efron_simulation(
            R=R,
            n=n,
            seed=seed + n,
            progress_every=max(1, R // 10),
            progress_callback=progress_callback,
            rf_n_estimators=rf_n_estimators,
        )
        rows.append(df_n)
        if progress_callback:
            progress_callback(
                f"Finished sample-size study n={n}: "
                f"{len(df_n)} rows in {perf_counter() - start:.1f}s"
            )
    return pd.concat(rows, ignore_index=True)


def summarize_bias_by_n(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo bias and MSE by sample size."""
    rows = []
    for n, g in df.groupby("n"):
        truth = g["true_error"]

        for estimator in ["apparent_error", "efron_error", "loo_error"]:
            estimate = g[estimator]
            valid = estimate.notna() & truth.notna()
            error = estimate[valid] - truth[valid]

            rows.append(
                {
                    "n": n,
                    "estimator": estimator,
                    "bias": error.mean(),
                    "mse": np.mean(error**2),
                    "variance": estimate[valid].var(),
                }
            )
    return pd.DataFrame(rows)


def one_general_replication(
    n: int,
    dgp: str,
    model_name: str,
    rng: np.random.Generator,
    bootstrap_B: int = 100,
    compute_loo: bool = True,
    test_n: int = 5000,
    rf_n_estimators: int = 100,
):
    """Run one replication for a general DGP/model setting."""
    x, y, pi_true = generate_data(n, dgp, rng)

    if len(np.unique(y)) < 2:
        return None

    rep_seed = int(rng.integers(0, 2**32 - 1))
    model = make_model(model_name, seed=rep_seed, rf_n_estimators=rf_n_estimators)

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
        boot = bootstrap_corrected_error(model, x, y, B=bootstrap_B, seed=rep_seed)
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


def run_general_setting(
    n: int,
    dgp: str,
    model_name: str,
    R: int = 200,
    bootstrap_B: int = 100,
    compute_loo: bool = True,
    test_n: int = 5000,
    seed: int = 0,
    progress_every: int | None = None,
    progress_callback: ProgressCallback | None = None,
    rf_n_estimators: int = 100,
) -> pd.DataFrame:
    """Run Monte Carlo simulation for one sample-size/DGP/model setting."""
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < R:
        result = one_general_replication(
            n=n,
            dgp=dgp,
            model_name=model_name,
            rng=rng,
            bootstrap_B=bootstrap_B,
            compute_loo=compute_loo,
            test_n=test_n,
            rf_n_estimators=rf_n_estimators,
        )
        if result is not None:
            result["replication"] = len(rows)
            rows.append(result)
            if progress_callback and progress_every and len(rows) % progress_every == 0:
                progress_callback(f"{dgp}/{model_name}/n={n}: {len(rows)}/{R} replications")
            gc.collect()
    return pd.DataFrame(rows)


def should_compute_loo(model_name: str, compute_loo: bool, compute_loo_rf: bool) -> bool:
    """Decide whether to compute leave-one-out CV for this model."""
    if not compute_loo:
        return False
    if model_name == "rf" and not compute_loo_rf:
        return False
    return True


def train_vs_test_study(
    n: int,
    dgp: str,
    model_name: str,
    R: int = 500,
    n_test: int = 20000,
    seed: int = 0,
    rf_n_estimators: int = 100,
) -> pd.DataFrame:
    """Compare training error with independent test error."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(R):
        x_train, y_train, _ = generate_data(n, dgp, rng)
        if len(np.unique(y_train)) < 2:
            continue

        model = make_model(model_name, rf_n_estimators=rf_n_estimators)
        model.fit(x_train, y_train)
        train_error = np.mean(model.predict(x_train) != y_train)

        x_test, y_test, _ = generate_data(n_test, dgp, rng)
        test_error = np.mean(model.predict(x_test) != y_test)

        rows.append(
            {
                "replication": r,
                "train_error": train_error,
                "test_error": test_error,
                "generalization_gap": test_error - train_error,
            }
        )
    return pd.DataFrame(rows)
