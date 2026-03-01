"""
Faster-Whisper ASR service with singleton model lifecycle management.

This module provides a production-ready Automatic Speech Recognition (ASR) service
using the Faster-Whisper implementation. It implements a singleton pattern to ensure
the heavy Whisper model is loaded only once per process, regardless of the number
of requests or concurrent users.
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from app.config import settings
from app.transcribers.audio_transcriber import AudioTranscriber
from app.transcribers.transcription_result import TranscriptionResult


class WhisperTranscriber(AudioTranscriber):
    """
    Implementation of AudioTranscriber using Faster-Whisper.

    This transcriber provides speech-to-text functionality using the
    Faster-Whisper optimized implementation of OpenAI's Whisper models.
    It leverages the singleton model instance managed by the lifespan
    context for efficient resource usage.
    """

    def __init__(
            self,
            model_size: str = settings.MODEL_SIZE.value,
            device: str = settings.COMPUTE_DEVICE.value,
            quantization: str = settings.QUANTIZATION.value,
            vad_enabled: bool = settings.VAD_ENABLED,
    ):
        logger.info("Initializing WhisperTranscriber")
        logger.info(
            "Loading Whisper model | size={} device={} compute_type={}",
            model_size,
            device,
            quantization,
        )

        self._model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=quantization,
        )
        self._vad_enabled = vad_enabled

        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe speech audio using the Faster-Whisper model.

        This method implements the AudioTranscriber interface, converting
        raw audio waveform into structured transcription results. It applies
        optional VAD filtering and returns both the transcribed text and
        language information.

        Parameters
        ----------
        audio : np.ndarray
            Pre-processed mono audio waveform. Expected format:
            - Sample rate: 16 kHz (min required by Whisper)
            - Data type: float32
            - Shape: (n_samples,) for mono audio

        Returns
        -------
        TranscriptionResult
            Structured result containing:
            - text: Concatenated transcription from all segments
            - language: Detected language code (ISO 639-1)
            - language_probability: Confidence score (0.0 to 1.0)
        """
        # Run transcription with optional VAD filtering
        segments, info = self._model.transcribe(
            audio,
            vad_filter=self._vad_enabled,
        )

        # Combine all segments into complete transcript
        hypothesis_text = "".join([segment.text for segment in segments])

        # Create structured result
        result = TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            text=hypothesis_text,
        )

        return result
