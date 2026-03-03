"""Immutable container for ASR output."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    """
    Holds the complete output of a single transcription run.

    Attributes
    ----------
    language : str
        ISO 639-1 code of the detected language (e.g. "en", "fr").
    language_probability : float
        Confidence of the language detection in [0.0, 1.0].
    text : str
        Full transcribed text. Empty string if no speech was detected.
    """

    language: str
    language_probability: float
    text: str
