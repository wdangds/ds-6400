"""Data-generating processes used in the course simulations."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def generate_efron_dgp(n: int, rng: np.random.Generator):
    """Generate Efron's sampling experiment data."""
    y = rng.binomial(n=1, p=0.5, size=n)
    x1 = rng.normal(loc=y - 0.5, scale=1, size=n)
    x2 = rng.normal(loc=0, scale=1, size=n)
    x = np.column_stack((x1, x2))
    pi_true = expit(x1)
    return x, y, pi_true


def generate_logistic_dgp(n: int, rng: np.random.Generator):
    """Generate a logistic regression data generating process."""
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = -0.2 + x1 - 0.7 * x2
    pi_true = expit(eta)
    y = rng.binomial(1, pi_true)
    x = np.column_stack([x1, x2])
    return x, y, pi_true


def generate_nonlinear_dgp(n: int, rng: np.random.Generator):
    """Generate a nonlinear logistic regression data generating process."""
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    eta = -0.2 + x1 - 0.7 * x2 + 1.1 * (x1**2 - 1)
    pi_true = expit(eta)
    y = rng.binomial(1, pi_true)
    x = np.column_stack([x1, x2])
    return x, y, pi_true


def generate_data(n: int, dgp: str, rng: np.random.Generator):
    """Generate data from a named data-generating process."""
    if dgp == "efron":
        return generate_efron_dgp(n, rng)
    if dgp == "logistic":
        return generate_logistic_dgp(n, rng)
    if dgp == "nonlinear":
        return generate_nonlinear_dgp(n, rng)
    raise ValueError(f"Unknown DGP: {dgp}")
