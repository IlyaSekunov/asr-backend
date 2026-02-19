"""
Automatic Speech Recognition service backed by Faster-Whisper.

Provides a lazily-initialised singleton transcriber so the heavy model is
loaded only once per process, regardless of how many requests arrive.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
from fastapi import FastAPI
from faster_whisper import WhisperModel
from loguru import logger

from app.core.config import settings


@dataclass(frozen=True)
class TranscriptionResult:
    language: str
    language_probability: float
    text: str


asr_whisper = {}


def get_asr_whisper() -> WhisperModel:
    if "asr_whisper" not in asr_whisper:
        raise Exception("ASR Whisper has not been loaded yet")

    return asr_whisper["asr_whisper"]


@asynccontextmanager
async def asr_whisper_lifespan(app: FastAPI):
    logger.info(
        "Loading Whisper model | size={} device={} compute_type={}",
        settings.MODEL_SIZE.value,
        settings.COMPUTE_DEVICE.value,
        settings.QUANTIZATION.value,
    )
    model = WhisperModel(
        settings.MODEL_SIZE.value,
        device=settings.COMPUTE_DEVICE.value,
        compute_type=settings.QUANTIZATION.value,
    )
    logger.info("Whisper model loaded successfully.")

    asr_whisper["asr_whisper"] = model

    yield


def transcribe(audio: np.ndarray) -> TranscriptionResult:
    """
    Run ASR on a pre-processed audio signal and return a structured result.

    The function respects the ``VAD_ENABLED`` flag in settings to control
    whether Silero VAD is applied before decoding.

    Parameters
    ----------
    audio:
        Pre-processed mono float32 waveform (expected at 16 kHz).

    Returns
    -------
    TranscriptionResult
        Structured transcription containing per-segment timing and text.
    """
    model = get_asr_whisper()

    logger.debug("Starting transcription")

    segments, info = model.transcribe(
        audio,
        vad_filter=settings.VAD_ENABLED,
    )

    hypothesis_text = "".join([segment.text for segment in segments])

    result = TranscriptionResult(
        language=info.language,
        language_probability=info.language_probability,
        text=hypothesis_text,
    )

    logger.info(
        "Transcription complete | language={} retrieved_text={}",
        result.language,
        result.text[:100] + "...",
    )
    return result
