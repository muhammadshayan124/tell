"""Cross-validated evaluation of the combined text+audio feature set.

Deliberately a plain, standardized logistic regression, not a deep model:
with ~120 labeled examples, anything more expressive would just memorize
noise and report a misleadingly high number. Logistic regression also keeps
the per-feature coefficients directly interpretable, which matters more
here than squeezing out another percentage point of accuracy -- the point
of this project is "which documented cues actually correlate, and by how
much," not "get the best possible score."

Stratified K-fold (not a single train/test split) is used because with this
few examples, a single split's accuracy is mostly noise -- the reported
number is the mean and standard deviation across folds, which is the
honest way to say "cross-validated accuracy was X +/- Y," not "accuracy
was X."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from tell.dataset import FEATURE_NAMES, LabeledExample

DEFAULT_N_FOLDS = 10
RANDOM_STATE = 42


@dataclass(frozen=True)
class EvaluationResult:
    n_examples: int
    n_deceptive: int
    n_truthful: int
    accuracy_mean: float
    accuracy_std: float
    fold_accuracies: list[float]
    precision: float
    recall: float
    f1: float
    feature_coefficients: dict[str, float]


def evaluate(examples: list[LabeledExample], n_folds: int = DEFAULT_N_FOLDS) -> EvaluationResult:
    X = np.array([e.feature_vector for e in examples])
    y = np.array([e.label for e in examples])

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []
    fold_coefficients = []

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_train, y[train_idx])
        fold_accuracies.append(clf.score(X_test, y[test_idx]))
        fold_coefficients.append(clf.coef_[0])

    y_pred = cross_val_predict(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X, y, cv=cv
    )
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    mean_coefficients = np.mean(fold_coefficients, axis=0)

    return EvaluationResult(
        n_examples=len(examples),
        n_deceptive=int(np.sum(y == 1)),
        n_truthful=int(np.sum(y == 0)),
        accuracy_mean=float(np.mean(fold_accuracies)),
        accuracy_std=float(np.std(fold_accuracies)),
        fold_accuracies=[float(a) for a in fold_accuracies],
        precision=precision,
        recall=recall,
        f1=f1,
        feature_coefficients=dict(
            zip(FEATURE_NAMES, (float(c) for c in mean_coefficients), strict=True)
        ),
    )
