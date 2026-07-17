"""Local speech-to-text via faster-whisper, so the GUI can go straight from a
microphone recording to a transcript without a cloud API call. Lazily
imported (same pattern as factcheck.py's anthropic import) since it's an
optional, fairly heavy dependency (~150MB+ model download on first use) that
most of this project's tests and CLI usage don't need.

Whisper is a transcription model, not a deception-cue model -- nothing about
its output is evidence of truthfulness one way or the other. It just gets
spoken words into text so the (separately weak, separately disclaimed)
text features in features/text.py have something to run on.
"""

from __future__ import annotations


class TranscriptionUnavailableError(RuntimeError):
    """Raised when faster-whisper is not installed."""


def transcribe(audio_path: str, model_size: str = "base") -> str:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionUnavailableError(
            "faster-whisper is not installed. Install it with: pip install 'tell[whisper]'"
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments).strip()
