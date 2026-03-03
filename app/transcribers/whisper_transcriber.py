"""Faster-Whisper ASR transcriber with lazy singleton model loading."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger

from app.config import settings
from app.transcribers.audio_transcriber import AudioTranscriber
from app.transcribers.transcription_result import TranscriptionResult


class WhisperTranscriber(AudioTranscriber):
    """
    AudioTranscriber implementation backed by Faster-Whisper.

    The WhisperModel is loaded once on construction and reused across all
    calls, so instantiate this class once per process.
    """

    def __init__(
            self,
            model_size: str = settings.MODEL_SIZE.value,
            device: str = settings.COMPUTE_DEVICE.value,
            quantization: str = settings.QUANTIZATION.value,
            vad_enabled: bool = settings.VAD_ENABLED,
    ):
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
        Transcribe a mono float32 waveform sampled at 16 kHz.

        Parameters
        ----------
        audio : np.ndarray
            Shape (n_samples,), dtype float32, sample rate 16 kHz.

        Returns
        -------
        TranscriptionResult
            Transcribed text, detected language code, and language confidence.
        """
        segments, info = self._model.transcribe(audio, vad_filter=self._vad_enabled)

        text = "".join(segment.text for segment in segments)

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            text=text,
        )
