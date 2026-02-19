"""
Audio loading and pre-processing pipeline.

Handles decoding of uploaded audio files and orchestrates the configurable
chain of pre-processing steps (denoising → loudness normalization) before
the signal is handed off to the ASR service.
"""

from __future__ import annotations

import os

import librosa
import numpy as np
from loguru import logger

from app.core.config import settings
from app.preprocessing.denoise import denoise_audio
from app.preprocessing.loudness import normalize_loudness
from app.util.io import extract_filename


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Decode raw audio bytes into a mono float32 numpy array.

    The audio is resampled to ``settings.TARGET_SAMPLE_RATE`` automatically.

    Parameters
    ----------
    file_path:
        Absolute path to the uploaded audio file (.mp3 or .wav).

    Returns
    -------
    tuple[np.ndarray, int]
        ``(audio, sample_rate)`` — mono float32 waveform and its sample rate.
    """
    try:
        audio, sr = librosa.load(file_path, sr=settings.TARGET_SAMPLE_RATE)

        logger.debug(
            "Loaded audio | file={} duration={:.2f}s sr={}",
            extract_filename(file_path),
            len(audio) / sr,
            sr,
        )

        return audio, sr

    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def run_preprocessing_pipeline(
        audio: np.ndarray,
        sr: int,
) -> np.ndarray:
    """
    Execute the configured pre-processing steps on a raw audio signal.

    Steps (each individually togglable via settings):
        1. Noise reduction
        2. Loudness normalization

    Parameters
    ----------
    audio:
        Raw mono float32 waveform.
    sr:
        Sample rate of *audio*.

    Returns
    -------
    np.ndarray
        Pre-processed waveform, ready for ASR inference.
    """
    if settings.DENOISING_ENABLED:
        logger.debug("Running denoising step.")
        audio = denoise_audio(audio, sr)

    if settings.LOUDNESS_NORMALIZATION_ENABLED:
        logger.debug("Running loudness normalization step.")
        audio = normalize_loudness(audio, sr)

    return audio
