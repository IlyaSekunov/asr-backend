"""
Faster-Whisper ASR service with singleton model lifecycle management.

This module provides a production-ready Automatic Speech Recognition (ASR) service
using the Faster-Whisper implementation. It implements a singleton pattern to ensure
the heavy Whisper model is loaded only once per process, regardless of the number
of requests or concurrent users.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI
from faster_whisper import WhisperModel
from loguru import logger

from app.transcribers.audio_transcriber import AudioTranscriber
from app.transcribers.transcription_result import TranscriptionResult
from app.config import settings

# Global model cache for singleton pattern
# Using a dictionary allows for potential future extension to multiple models
_whisper_model: Dict[str, WhisperModel] = {}


def _get_asr_whisper() -> WhisperModel:
    """
    Retrieve the singleton Whisper model instance.

    This internal function provides access to the globally cached Whisper model.
    It ensures the model has been properly initialized before use, raising an
    exception if accessed before lifespan setup.

    Returns
    -------
    WhisperModel
        The loaded Faster-Whisper model instance.

    Raises
    ------
    RuntimeError
        If the model hasn't been loaded yet via the lifespan context.
        This indicates a lifecycle management issue in the application.
    """
    if "whisper_model" not in _whisper_model:
        error_msg = (
            "ASR Whisper model accessed before initialization. "
            "Ensure asr_whisper_lifespan context manager is properly configured."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return _whisper_model["whisper_model"]


@asynccontextmanager
async def asr_whisper_lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for Whisper model lifecycle.

    This context manager handles the complete lifecycle of the Whisper model:
        - Startup: Loads the model with configured parameters
        - Runtime: Keeps model available for requests
        - Shutdown: (Optional) Cleans up resources when application stops

    The model is loaded once at application startup and shared across all
    requests, preventing repeated expensive loading operations.
    """
    logger.info(
        "Loading Whisper model | size={} device={} compute_type={}",
        settings.MODEL_SIZE.value,
        settings.COMPUTE_DEVICE.value,
        settings.QUANTIZATION.value,
    )

    try:
        model = WhisperModel(
            settings.MODEL_SIZE.value,
            device=settings.COMPUTE_DEVICE.value,
            compute_type=settings.QUANTIZATION.value,
        )
        _whisper_model["whisper_model"] = model
        logger.info("Whisper model loaded successfully.")

        yield

    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise
    finally:
        # Optional cleanup can be added here if needed
        # e.g., model unload, GPU memory cleanup
        logger.debug("Whisper model lifecycle ended.")


class WhisperTranscriber(AudioTranscriber):
    """
    Implementation of AudioTranscriber using Faster-Whisper.

    This transcriber provides speech-to-text functionality using the
    Faster-Whisper optimized implementation of OpenAI's Whisper models.
    It leverages the singleton model instance managed by the lifespan
    context for efficient resource usage.
    """

    def __init__(self, vad_enabled: bool = settings.VAD_ENABLED):
        self._model = None
        self.vad_enabled = vad_enabled

    @property
    def model(self) -> WhisperModel:
        """
        Lazy-load the Whisper model when first accessed.

        Returns
        -------
        WhisperModel
            The singleton Whisper model instance.

        Raises
        ------
        RuntimeError
            If model hasn't been loaded via lifespan context.
        """
        if self._model is None:
            self._model = _get_asr_whisper()
        return self._model

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
        segments, info = self.model.transcribe(
            audio,
            vad_filter=settings.VAD_ENABLED,
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
