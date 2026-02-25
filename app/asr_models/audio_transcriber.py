from abc import ABC, abstractmethod

import numpy as np

from app.asr_models.transcription_result import TranscriptionResult


class AudioTranscriber(ABC):
    """
    Abstract base class for automatic speech recognition (ASR) models.

    Defines the interface for all audio transcription implementations in the system.
    Concrete subclasses must implement the transcribe() method to convert audio
    signals into text using various ASR backends (e.g., Whisper, Wav2Vec2).

    The class follows the Strategy pattern, allowing different transcription
    engines to be swapped seamlessly while maintaining a consistent interface
    for the rest of the application.

    See Also
    --------
    WhisperTranscriber : Implementation using OpenAI's Whisper model
    """

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Convert speech audio to text using the ASR model.

        Parameters
        ----------
        audio : np.ndarray
            Audio time series to transcribe. Expected format:
            - Values normalized to [-1.0, 1.0] range
            - Shape can be (n_samples,) for mono or (n_channels, n_samples)
              for multi-channel audio
            - dtype should be float32 or float64
            - Sample rate is implementation-dependent (should be documented
              by concrete classes)

        Returns
        -------
        TranscriptionResult
            Immutable container with:
            - Transcribed text (empty string if no speech detected)
            - Detected language (ISO 639-1 code)
            - Language detection confidence (0.0 to 1.0)

        Raises
        ------
        ValueError
            If the input audio format is invalid (wrong dimensions,
            unsupported sample rate, empty array, etc.)
        RuntimeError
            If transcription fails due to model errors, GPU issues,
            or other runtime problems
        NotImplementedError
            If called directly on the base class
        """
        pass
