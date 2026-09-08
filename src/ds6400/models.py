"""Model factories used by the simulation studies."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def make_model(
    model_name: str,
    seed: int | None = None,
    rf_n_estimators: int = 100,
    n_jobs: int | None = 1,
):
    """Construct a prediction model by name."""
    if model_name == "logistic":
        return LogisticRegression(C=np.inf, solver="lbfgs", random_state=seed)
    if model_name == "lda":
        return LinearDiscriminantAnalysis()
    if model_name == "1nn":
        return KNeighborsClassifier(n_neighbors=1)
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=None,
            random_state=seed,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model name: {model_name}")
