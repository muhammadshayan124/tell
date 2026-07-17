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
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
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


def fit_final_model(examples: list[LabeledExample]) -> Pipeline:
    """Fits one Pipeline(StandardScaler, LogisticRegression) on ALL examples.

    This is the model actually saved and used for inference -- it is NOT the
    same thing as the cross-validated `evaluate()` result above, and its
    training accuracy means nothing on its own. The honest generalization
    estimate is, and always will be, the cross-validated number from
    `evaluate()`; this function exists only to produce a single deployable
    artifact from the same features and the same modeling choice.
    """
    X = np.array([e.feature_vector for e in examples])
    y = np.array([e.label for e in examples])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    import skops.io as sio

    sio.dump(pipeline, path)


def load_model(path: str | Path) -> Pipeline:
    import skops.io as sio

    trusted = sio.get_untrusted_types(file=path)
    return sio.load(path, trusted=trusted)


@dataclass(frozen=True)
class FeatureContribution:
    name: str
    raw_value: float
    z_score: float
    coefficient: float
    contribution_to_logit: float


def explain(pipeline: Pipeline, feature_vector: list[float]) -> list[FeatureContribution]:
    """Breaks a single prediction down into per-feature terms of the
    underlying logistic regression: contribution_to_logit = z_score * coefficient.

    Summing every contribution plus the intercept gives the model's raw
    logit for this example; passing that through a sigmoid gives the
    probability the model reports. This function exists so a human can see
    that arithmetic explicitly, rather than being handed an opaque score.
    """
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    x = np.array(feature_vector)
    z_scores = (x - scaler.mean_) / scaler.scale_
    coefficients = clf.coef_[0]
    contributions = z_scores * coefficients

    return [
        FeatureContribution(
            name=name,
            raw_value=float(x[i]),
            z_score=float(z_scores[i]),
            coefficient=float(coefficients[i]),
            contribution_to_logit=float(contributions[i]),
        )
        for i, name in enumerate(FEATURE_NAMES)
    ]
