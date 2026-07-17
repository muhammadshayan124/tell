import random

from tell.dataset import FEATURE_NAMES, LabeledExample
from tell.model import evaluate


def _make_separable_examples(n_per_class: int = 20) -> list[LabeledExample]:
    """Two clearly separated Gaussian blobs in feature space -- a sanity
    check that the pipeline (scaling, CV, scoring) works end-to-end and can
    recover a strong signal when one genuinely exists, independent of
    whether real deception cues are actually that strong."""
    rng = random.Random(0)
    n_features = len(FEATURE_NAMES)
    examples = []
    for label, center in ((0, 0.0), (1, 10.0)):
        for i in range(n_per_class):
            vector = [center + rng.gauss(0, 0.5) for _ in range(n_features)]
            examples.append(
                LabeledExample(clip_id=f"{label}_{i}", label=label, feature_vector=vector)
            )
    return examples


def test_evaluate_recovers_a_strong_synthetic_signal():
    examples = _make_separable_examples()
    result = evaluate(examples, n_folds=5)
    assert result.accuracy_mean > 0.9


def test_evaluate_reports_correct_class_counts():
    examples = _make_separable_examples(n_per_class=15)
    result = evaluate(examples, n_folds=5)
    assert result.n_examples == 30
    assert result.n_deceptive == 15
    assert result.n_truthful == 15


def test_evaluate_returns_one_accuracy_per_fold():
    examples = _make_separable_examples()
    result = evaluate(examples, n_folds=4)
    assert len(result.fold_accuracies) == 4


def test_evaluate_feature_coefficients_cover_every_named_feature():
    examples = _make_separable_examples()
    result = evaluate(examples, n_folds=5)
    assert set(result.feature_coefficients.keys()) == set(FEATURE_NAMES)


def test_evaluate_on_random_labels_is_close_to_chance():
    """The flip side of the separable-signal test: features that carry no
    real information about the label should NOT produce a confidently high
    accuracy -- if this test failed, it would mean the evaluation harness
    itself has a leak (e.g. label information bleeding into features)."""
    rng = random.Random(1)
    n_features = len(FEATURE_NAMES)
    examples = [
        LabeledExample(
            clip_id=str(i),
            label=rng.randint(0, 1),
            feature_vector=[rng.gauss(0, 1) for _ in range(n_features)],
        )
        for i in range(80)
    ]
    result = evaluate(examples, n_folds=8)
    assert 0.3 < result.accuracy_mean < 0.7
