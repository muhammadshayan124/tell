"""Builds features from the prepared dataset, runs the honest cross-validated
evaluation, fits a final deployable model on all examples, and saves both
the model artifact and a model card documenting exactly what it is and isn't.

Usage:
  python scripts/train_and_save_model.py --data-dir data/Real-life_Deception_Detection_2016 \
      --out models/
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tell.dataset import build_examples
from tell.model import evaluate, fit_final_model, save_model

MODEL_CARD_TEMPLATE = """---
license: mit
tags:
  - deception-detection
  - research-artifact
  - not-for-production-decisions
---

# tell -- experimental deception-cue classifier

**This is not a validated lie detector.**
See the [project README](https://github.com/muhammadshayan124/tell) for the full disclaimer.
This model card exists so anyone loading this model from the Hugging Face Hub sees the same
caveats as everyone using the CLI/GUI directly.

## What this model is

A logistic regression over {n_features} documented linguistic + vocal features (see README for
citations), fit on {n_examples} real, labeled courtroom clips ({n_deceptive} deceptive,
{n_truthful} truthful) from the [Real-life Trial Deception Detection dataset][dataset]
(Perez-Rosas et al., 2015).

[dataset]: https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip

## Honest performance (10-fold cross-validation, NOT training accuracy)

- Accuracy: {accuracy_mean:.1%} +/- {accuracy_std:.1%}
- Precision: {precision:.1%}
- Recall: {recall:.1%}
- F1: {f1:.1%}

Chance is 50%. This is a real, weak, above-chance signal, consistent with the published
literature's own conclusion that these cues are individually weak and inconsistent. It is NOT
evidence this model can tell whether a specific person is lying.

## Known limitations

- Labels are trial verdicts, not verified ground truth about whether the person lied.
- Many clips share the same speaker/case, so the model may partly be learning to recognize
  who is talking rather than a generalizable cue -- cross-validation folds are not guaranteed
  speaker-disjoint.
- n=121 is small; the +/- {accuracy_std:.1%} standard deviation across folds should be read
  as "we genuinely don't know the true accuracy any more precisely than this," not a rounding
  error.

## Do not use this to make a real decision about a real person.

Generated {generated_at} from a fixed random seed; see the `tell` repository for the full pipeline.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/Real-life_Deception_Detection_2016",
        help="Prepared dataset directory.",
    )
    parser.add_argument(
        "--out", default="models", help="Output directory for the saved model + model card."
    )
    parser.add_argument("--folds", type=int, default=10, help="Number of cross-validation folds.")
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and extracting features from {dataset_dir} ...")
    examples = build_examples(dataset_dir)
    print(f"Loaded {len(examples)} labeled examples.")

    print(f"Running honest {args.folds}-fold cross-validation ...")
    result = evaluate(examples, n_folds=args.folds)
    print(f"Cross-validated accuracy: {result.accuracy_mean:.1%} +/- {result.accuracy_std:.1%}")
    print(f"Precision: {result.precision:.1%}  Recall: {result.recall:.1%}  F1: {result.f1:.1%}")

    print("Fitting the final deployable model on all examples ...")
    pipeline = fit_final_model(examples)

    model_path = out_dir / "tell_classifier.skops"
    save_model(pipeline, model_path)
    print(f"Saved model to {model_path}")

    metadata = {
        "n_examples": result.n_examples,
        "n_deceptive": result.n_deceptive,
        "n_truthful": result.n_truthful,
        "cross_validated_accuracy_mean": result.accuracy_mean,
        "cross_validated_accuracy_std": result.accuracy_std,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "feature_coefficients": result.feature_coefficients,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata to {metadata_path}")

    model_card = MODEL_CARD_TEMPLATE.format(
        n_features=len(result.feature_coefficients),
        n_examples=result.n_examples,
        n_deceptive=result.n_deceptive,
        n_truthful=result.n_truthful,
        accuracy_mean=result.accuracy_mean,
        accuracy_std=result.accuracy_std,
        precision=result.precision,
        recall=result.recall,
        f1=result.f1,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    card_path = out_dir / "model_card.md"
    card_path.write_text(model_card, encoding="utf-8")
    print(f"Saved model card to {card_path}")


if __name__ == "__main__":
    main()
