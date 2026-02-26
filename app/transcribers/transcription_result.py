from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    """
    Immutable container for automatic speech recognition (ASR) output.

    This dataclass encapsulates the complete result of a speech-to-text
    transcription, including the transcribed text, detected language,
    and confidence metrics.

    Attributes
    ----------
    language : str
        ISO 639-1 language code (e.g., 'en', 'fr', 'de') or full language
        name detected in the audio. Represents the primary language used
        in the transcribed speech.

    language_probability : float
        Confidence score for the detected language, ranging from 0.0 to 1.0.
        Higher values indicate greater confidence in the language detection.
        Values near 1.0 suggest high certainty, while values near 0.0
        indicate ambiguity or potential misdetection.

    text : str
        The transcribed text content from the audio source. May contain
        punctuation, capitalization, and formatting depending on the ASR
        model configuration. Empty string indicates no speech detected
        or failed transcription.
    """

    language: str
    """ISO 639-1 language code or name of detected language."""

    language_probability: float
    """Confidence score (0.0-1.0) for language detection."""

    text: str
    """Transcribed text content from the audio source."""
