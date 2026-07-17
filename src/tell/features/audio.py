"""Vocal prosody features -- read the module docstring in text.py first for
the general framing; the short version for audio specifically is that
published effect sizes for vocal/paraverbal cues to deception are, if
anything, weaker and less consistent than the linguistic ones (Sporer &
Schwandt, 2006, "Paraverbal indicators of deception: A meta-analytic
synthesis," Applied Cognitive Psychology, 20(4), 421-446 -- most cues here
had effect sizes indistinguishable from zero across studies). These features
exist to see what a real, honest evaluation says, not because the
literature promises they'll work.

Features:
  - Pitch (F0) mean/std via librosa's pyin (probabilistic YIN) -- vocal
    tension is the classic (if weakly supported) hypothesis here.
  - Pause ratio and pause count via librosa's silence-split, which is also
    just a reasonable proxy for speech rate without needing a full ASR
    forced-alignment pipeline.
  - RMS energy mean/std -- vocal intensity and its variability.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

FMIN_HZ = 65.0  # below typical adult male F0 floor
FMAX_HZ = 400.0  # above typical adult female F0 ceiling
SILENCE_TOP_DB = 30  # frames quieter than (peak - this) count as a pause


@dataclass(frozen=True)
class AudioFeatures:
    duration_seconds: float
    pitch_mean_hz: float
    pitch_std_hz: float
    voiced_ratio: float
    pause_count: int
    pause_ratio: float
    rms_mean: float
    rms_std: float


def extract_audio_features(path: str) -> AudioFeatures:
    y, sr = librosa.load(path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    f0, voiced_flag, _ = librosa.pyin(y, fmin=FMIN_HZ, fmax=FMAX_HZ, sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = float(np.nanmean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std = float(np.nanstd(voiced_f0)) if voiced_f0.size else 0.0
    voiced_ratio = (
        float(np.sum(voiced_flag)) / len(voiced_flag)
        if voiced_flag is not None and len(voiced_flag)
        else 0.0
    )

    non_silent_intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB)
    non_silent_seconds = sum((end - start) / sr for start, end in non_silent_intervals)
    pause_ratio = 1.0 - (non_silent_seconds / duration if duration > 0 else 0.0)
    # A "pause" is a gap between two consecutive voiced segments, not the
    # segments themselves -- len(intervals) - 1 gaps for N segments.
    pause_count = max(len(non_silent_intervals) - 1, 0)

    rms = librosa.feature.rms(y=y)[0]

    return AudioFeatures(
        duration_seconds=float(duration),
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
        voiced_ratio=voiced_ratio,
        pause_count=pause_count,
        pause_ratio=float(pause_ratio),
        rms_mean=float(np.mean(rms)),
        rms_std=float(np.std(rms)),
    )


FEATURE_NAMES = [
    "duration_seconds",
    "pitch_mean_hz",
    "pitch_std_hz",
    "voiced_ratio",
    "pause_count",
    "pause_ratio",
    "rms_mean",
    "rms_std",
]


def features_to_vector(features: AudioFeatures) -> list[float]:
    return [getattr(features, name) for name in FEATURE_NAMES]
