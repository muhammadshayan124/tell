import csv

import numpy as np
import soundfile as sf

from tell.dataset import FEATURE_NAMES, build_examples, load_manifest


def _make_tiny_dataset(tmp_path):
    (tmp_path / "Transcription" / "Deceptive").mkdir(parents=True)
    (tmp_path / "Transcription" / "Truthful").mkdir(parents=True)
    (tmp_path / "Audio" / "Deceptive").mkdir(parents=True)
    (tmp_path / "Audio" / "Truthful").mkdir(parents=True)

    (tmp_path / "Transcription" / "Deceptive" / "clip_lie_1.txt").write_text("I did not do it")
    (tmp_path / "Transcription" / "Truthful" / "clip_truth_1.txt").write_text("I was at home")

    sr = 16000
    silence = np.zeros(sr)
    sf.write(str(tmp_path / "Audio" / "Deceptive" / "clip_lie_1.wav"), silence, sr)
    sf.write(str(tmp_path / "Audio" / "Truthful" / "clip_truth_1.wav"), silence, sr)

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "label", "transcript_path", "audio_path"])
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "clip_lie_1",
                "label": 1,
                "transcript_path": "Transcription/Deceptive/clip_lie_1.txt",
                "audio_path": "Audio/Deceptive/clip_lie_1.wav",
            }
        )
        writer.writerow(
            {
                "clip_id": "clip_truth_1",
                "label": 0,
                "transcript_path": "Transcription/Truthful/clip_truth_1.txt",
                "audio_path": "Audio/Truthful/clip_truth_1.wav",
            }
        )
    return manifest_path


def test_load_manifest_reads_all_rows(tmp_path):
    manifest_path = _make_tiny_dataset(tmp_path)
    rows = load_manifest(manifest_path)
    assert len(rows) == 2
    assert {row["clip_id"] for row in rows} == {"clip_lie_1", "clip_truth_1"}


def test_build_examples_produces_one_example_per_manifest_row(tmp_path):
    _make_tiny_dataset(tmp_path)
    examples = build_examples(tmp_path)
    assert len(examples) == 2
    assert {e.clip_id for e in examples} == {"clip_lie_1", "clip_truth_1"}


def test_build_examples_preserves_labels(tmp_path):
    _make_tiny_dataset(tmp_path)
    examples = {e.clip_id: e for e in build_examples(tmp_path)}
    assert examples["clip_lie_1"].label == 1
    assert examples["clip_truth_1"].label == 0


def test_build_examples_feature_vector_length_matches_combined_feature_names(tmp_path):
    _make_tiny_dataset(tmp_path)
    examples = build_examples(tmp_path)
    assert all(len(e.feature_vector) == len(FEATURE_NAMES) for e in examples)
