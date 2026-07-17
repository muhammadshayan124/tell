from __future__ import annotations

import json
from pathlib import Path

import click

from tell import model as model_module
from tell.dataset import build_examples
from tell.features import audio as audio_features
from tell.features import text as text_features

DISCLAIMER = (
    "tell is an experimental research artifact, not a validated lie detector. "
    "See README.md. Do not use this to make a real decision about a real person."
)


@click.group()
def main() -> None:
    """tell -- an honest experiment in combining documented deception cues.

    \b
    """
    click.echo(DISCLAIMER, err=True)


@main.command()
@click.option(
    "--data-dir",
    default="data/Real-life_Deception_Detection_2016",
    help="Prepared dataset directory (see scripts/fetch_dataset.py).",
)
@click.option(
    "--folds", default=model_module.DEFAULT_N_FOLDS, help="Number of cross-validation folds."
)
def evaluate(data_dir: str, folds: int) -> None:
    """Run cross-validated evaluation of the cue-based model on the labeled dataset."""
    dataset_dir = Path(data_dir)
    click.echo(f"Loading and extracting features from {dataset_dir} ...")
    examples = build_examples(dataset_dir)
    click.echo(
        f"Loaded {len(examples)} labeled examples. Running {folds}-fold cross-validation ..."
    )
    result = model_module.evaluate(examples, n_folds=folds)

    click.echo("")
    click.echo(
        f"n = {result.n_examples} ({result.n_deceptive} deceptive, {result.n_truthful} truthful)"
    )
    click.echo(
        f"Cross-validated accuracy: {result.accuracy_mean:.1%} +/- {result.accuracy_std:.1%}"
    )
    click.echo(
        f"Precision: {result.precision:.1%}  Recall: {result.recall:.1%}  F1: {result.f1:.1%}"
    )
    click.echo("")
    click.echo("Feature coefficients (mean across folds, standardized features):")
    for name, coef in sorted(result.feature_coefficients.items(), key=lambda kv: -abs(kv[1])):
        click.echo(f"  {name:>30s}  {coef:+.3f}")


@main.command()
@click.argument("transcript_file", type=click.Path(exists=True))
@click.argument("audio_file", type=click.Path(exists=True))
def analyze(transcript_file: str, audio_file: str) -> None:
    """Extract and print the raw cue features for one transcript+audio pair.

    This intentionally does NOT print a "lie probability" -- see README for
    why a single confidence number would misrepresent what this model can
    actually tell you.
    """
    transcript = Path(transcript_file).read_text(encoding="utf-8")
    text_feats = text_features.extract_text_features(transcript)
    audio_feats = audio_features.extract_audio_features(audio_file)

    click.echo(
        json.dumps(
            {"text_features": text_feats.__dict__, "audio_features": audio_feats.__dict__}, indent=2
        )
    )


if __name__ == "__main__":
    main()
