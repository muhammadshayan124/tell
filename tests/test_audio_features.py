import numpy as np
import soundfile as sf

from tell.features.audio import extract_audio_features


def _write_wav(path, samples, sr=16000):
    sf.write(str(path), samples, sr)


def test_pitch_mean_matches_a_known_pure_tone(tmp_path):
    sr = 16000
    duration = 1.0
    freq = 150.0  # within FMIN_HZ..FMAX_HZ
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)

    path = tmp_path / "tone.wav"
    _write_wav(path, tone, sr)

    features = extract_audio_features(str(path))
    # pyin's pitch estimate on a pure tone should land close to the true frequency.
    assert abs(features.pitch_mean_hz - freq) < 10.0
    assert features.voiced_ratio > 0.5


def test_pure_silence_or_uniform_quiet_has_zero_voiced_ratio(tmp_path):
    sr = 16000
    silence = np.zeros(sr)
    path = tmp_path / "silence.wav"
    _write_wav(path, silence, sr)

    features = extract_audio_features(str(path))
    # pyin correctly finds no pitch in silence regardless of the split quirk below.
    assert features.voiced_ratio == 0.0
    assert features.pitch_mean_hz == 0.0


def test_pause_detection_finds_the_quiet_gap_between_two_tones(tmp_path):
    # librosa.effects.split's dB threshold is relative to the signal's OWN
    # peak, so it only finds pauses *within* a clip that has both loud and
    # quiet parts to contrast -- a uniformly quiet (or uniformly loud) clip
    # has no internal contrast to split on, so testing on silence alone
    # would be testing the wrong thing. This mimics real speech instead:
    # tone - gap - tone, and checks the gap is correctly recovered as a pause.
    sr = 16000
    tone_seconds = 0.4
    gap_seconds = 0.4
    t = np.linspace(0, tone_seconds, int(sr * tone_seconds), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 200 * t)
    gap = np.zeros(int(sr * gap_seconds))

    samples = np.concatenate([tone, gap, tone])
    path = tmp_path / "tone_gap_tone.wav"
    _write_wav(path, samples, sr)

    features = extract_audio_features(str(path))
    assert features.pause_count >= 1
    # The gap is roughly a third of the total duration.
    assert 0.2 < features.pause_ratio < 0.5


def test_duration_matches_file_length(tmp_path):
    sr = 16000
    samples = np.zeros(sr * 2)  # 2 seconds
    path = tmp_path / "two_seconds.wav"
    _write_wav(path, samples, sr)

    features = extract_audio_features(str(path))
    assert abs(features.duration_seconds - 2.0) < 0.05


def test_louder_signal_has_higher_rms_than_quieter_signal(tmp_path):
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    loud = 0.8 * np.sin(2 * np.pi * 200 * t)
    quiet = 0.05 * np.sin(2 * np.pi * 200 * t)

    loud_path, quiet_path = tmp_path / "loud.wav", tmp_path / "quiet.wav"
    _write_wav(loud_path, loud, sr)
    _write_wav(quiet_path, quiet, sr)

    loud_features = extract_audio_features(str(loud_path))
    quiet_features = extract_audio_features(str(quiet_path))
    assert loud_features.rms_mean > quiet_features.rms_mean
