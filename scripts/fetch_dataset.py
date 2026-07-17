"""Downloads and prepares the Real-life Trial Deception Detection dataset
(Perez-Rosas et al., 2015) for local use. Not committed to this repo -- it's
248MB of third-party video and the license terms for redistribution aren't
stated on the source page, so this script re-fetches from the original
University of Michigan host rather than bundling a copy.

Citation (see README for the full bibtex):
  Perez-Rosas, V., Abouelenien, M., Mihalcea, R., & Burzo, M. (2015).
  "Deception Detection using Real-life Trial Data." ICMI '15.

Usage:
  python scripts/fetch_dataset.py [--out data/]
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import imageio_ffmpeg

DATASET_URL = "https://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip"
ARCHIVE_ROOT = "Real-life_Deception_Detection_2016"


def download(out_dir: Path) -> Path:
    zip_path = out_dir / "trial_deception.zip"
    if zip_path.exists():
        print(f"Already downloaded: {zip_path}")
        return zip_path
    print(f"Downloading {DATASET_URL} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    return zip_path


def extract(zip_path: Path, out_dir: Path) -> Path:
    dataset_dir = out_dir / ARCHIVE_ROOT
    if dataset_dir.exists():
        print(f"Already extracted: {dataset_dir}")
        return dataset_dir
    print(f"Extracting to {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        zf.extractall(out_dir, members=members)
    return dataset_dir


def extract_audio(dataset_dir: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for label in ("Deceptive", "Truthful"):
        clips_dir = dataset_dir / "Clips" / label
        audio_dir = dataset_dir / "Audio" / label
        audio_dir.mkdir(parents=True, exist_ok=True)
        for clip in sorted(clips_dir.glob("*.mp4")):
            wav_path = audio_dir / f"{clip.stem}.wav"
            if wav_path.exists():
                continue
            print(f"Extracting audio: {clip.name}")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(clip),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(wav_path),
                    "-loglevel",
                    "error",
                ],
                check=True,
            )


def build_manifest(dataset_dir: Path) -> Path:
    manifest_path = dataset_dir / "manifest.csv"
    rows = []
    for label, is_deceptive in (("Deceptive", 1), ("Truthful", 0)):
        transcript_dir = dataset_dir / "Transcription" / label
        audio_dir = dataset_dir / "Audio" / label
        for transcript_path in sorted(transcript_dir.glob("*.txt")):
            clip_id = transcript_path.stem
            audio_path = audio_dir / f"{clip_id}.wav"
            if not audio_path.exists():
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "label": is_deceptive,
                    "transcript_path": str(transcript_path.relative_to(dataset_dir)),
                    "audio_path": str(audio_path.relative_to(dataset_dir)),
                }
            )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "label", "transcript_path", "audio_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest with {len(rows)} rows to {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    zip_path = download(out_dir)
    dataset_dir = extract(zip_path, out_dir)
    extract_audio(dataset_dir)
    build_manifest(dataset_dir)
    zip_path.unlink(missing_ok=True)
    shutil.rmtree(out_dir / "__MACOSX", ignore_errors=True)


if __name__ == "__main__":
    main()
