"""Abstract base class for ASR transcribers."""

from abc import ABC, abstractmethod

import numpy as np

from app.transcribers.transcription_result import TranscriptionResult


class AudioTranscriber(ABC):
    """
    Interface for speech-to-text backends (e.g. Whisper, Wav2Vec2).

    Concrete implementations are expected to load their model on construction
    and expose transcription through the single `transcribe` method.
    """

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Convert a waveform to text.

        Parameters
        ----------
        audio : np.ndarray
            Mono float32 waveform. Sample rate must match the model's requirement.

        Returns
        -------
        TranscriptionResult
            Transcribed text, detected language code, and language confidence.

        Raises
        ------
        ValueError
            If the audio format is invalid.
        RuntimeError
            If the model fails during inference.
        """
        pass
