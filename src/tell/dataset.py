"""Loads the manifest produced by scripts/fetch_dataset.py and extracts the
combined text+audio feature vector for every labeled clip."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from tell.features import audio as audio_features
from tell.features import text as text_features


@dataclass(frozen=True)
class LabeledExample:
    clip_id: str
    label: int  # 1 = deceptive, 0 = truthful, per the dataset's verdict-based labeling
    feature_vector: list[float]


FEATURE_NAMES = text_features.FEATURE_NAMES + audio_features.FEATURE_NAMES


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with open(manifest_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_examples(dataset_dir: Path, manifest_path: Path | None = None) -> list[LabeledExample]:
    manifest_path = manifest_path or (dataset_dir / "manifest.csv")
    examples = []
    for row in load_manifest(manifest_path):
        transcript = (dataset_dir / row["transcript_path"]).read_text(encoding="utf-8")
        text_vec = text_features.features_to_vector(text_features.extract_text_features(transcript))
        audio_vec = audio_features.features_to_vector(
            audio_features.extract_audio_features(str(dataset_dir / row["audio_path"]))
        )
        examples.append(
            LabeledExample(
                clip_id=row["clip_id"],
                label=int(row["label"]),
                feature_vector=text_vec + audio_vec,
            )
        )
    return examples
