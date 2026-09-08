"""Prediction-error estimators and evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score

from ds6400.data_generation import generate_data


def apparent_error(model, x, y) -> float:
    """Compute the apparent error of a fitted model on the training data."""
    y_pred = model.predict(x)
    return float(np.mean(y_pred != y))


def true_fixed_x_error(model, x, pi_true) -> float:
    """Compute the true fixed-X error of a fitted classifier."""
    y_hat = model.predict(x)
    error_probability = np.where(y_hat == 1, 1 - pi_true, pi_true)
    return float(np.mean(error_probability))


def efron_optimism(model, x, cutoff: float = 0.5) -> float:
    """Compute Efron's analytic optimism estimate for logistic regression."""
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
    """Compute leave-one-out cross-validation error."""
    loo = LeaveOneOut()
    scores = cross_val_score(model, x, y, cv=loo, scoring="accuracy")
    return float(1 - scores.mean())


def kfold_cv_error(model, x, y, k: int = 10, seed: int = 0) -> float:
    """Compute stratified k-fold cross-validation error."""
    class_counts = np.bincount(y)
    if len(class_counts) < 2 or class_counts.min() < k:
        return np.nan

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(model), x, y, cv=cv, scoring="accuracy")
    return float(1 - scores.mean())


def bootstrap_corrected_error(model, x, y, B: int = 200, seed: int = 0) -> float:
    """Compute a generic nonparametric bootstrap optimism-corrected error."""
    rng = np.random.default_rng(seed)
    n = len(y)
    original_model = clone(model)
    original_model.fit(x, y)
    apparent = apparent_error(original_model, x, y)
    optimism_values = []

    for _ in range(B):
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


def independent_test_error(model, dgp: str, rng: np.random.Generator, n_test: int = 5000) -> float:
    """Evaluate a fitted model on new observations from the population."""
    x_test, y_test, _ = generate_data(n_test, dgp, rng)
    y_hat = model.predict(x_test)
    return float(np.mean(y_hat != y_test))
