import math
import random

from tell.dataset import FEATURE_NAMES, LabeledExample
from tell.model import explain, fit_final_model, load_model, save_model


def _make_separable_examples(n_per_class: int = 20) -> list[LabeledExample]:
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


def test_fit_final_model_predicts_the_training_data_it_was_fit_on():
    examples = _make_separable_examples()
    pipeline = fit_final_model(examples)
    X = [e.feature_vector for e in examples]
    y = [e.label for e in examples]
    assert pipeline.score(X, y) > 0.9


def test_save_and_load_model_roundtrips_predictions(tmp_path):
    examples = _make_separable_examples()
    pipeline = fit_final_model(examples)
    path = tmp_path / "model.skops"
    save_model(pipeline, path)
    loaded = load_model(path)

    sample = examples[0].feature_vector
    original_proba = pipeline.predict_proba([sample])[0]
    loaded_proba = loaded.predict_proba([sample])[0]
    assert list(original_proba) == list(loaded_proba)


def test_explain_covers_every_feature_and_sums_to_the_logit():
    examples = _make_separable_examples()
    pipeline = fit_final_model(examples)
    sample = examples[0].feature_vector

    contributions = explain(pipeline, sample)
    assert [c.name for c in contributions] == FEATURE_NAMES

    intercept = float(pipeline.named_steps["clf"].intercept_[0])
    logit = intercept + sum(c.contribution_to_logit for c in contributions)
    probability_from_contributions = 1.0 / (1.0 + math.exp(-logit))

    probability_from_model = pipeline.predict_proba([sample])[0][1]
    assert abs(probability_from_contributions - probability_from_model) < 1e-9


def test_explain_z_score_matches_manual_standardization():
    examples = _make_separable_examples()
    pipeline = fit_final_model(examples)
    sample = examples[0].feature_vector

    contributions = explain(pipeline, sample)
    scaler = pipeline.named_steps["scaler"]
    for i, c in enumerate(contributions):
        expected_z = (sample[i] - scaler.mean_[i]) / scaler.scale_[i]
        assert abs(c.z_score - expected_z) < 1e-9
