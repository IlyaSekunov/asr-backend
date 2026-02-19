"""
Loudness normalization utilities.

Exposes LUFS (ITU-R BS.1770), peak, and RMS normalization strategies, plus
helper functions for loudness measurement and silence detection.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from loguru import logger

from app.core.config import LoudnessNormalization, settings


def normalize_loudness(
        audio: np.ndarray,
        sr: int,
        target_loudness: float = settings.LOUDNESS_TARGET,
        method: LoudnessNormalization = settings.LOUDNESS_METHOD,
) -> np.ndarray:
    """
    Normalize the perceived loudness of an audio signal.

    Parameters
    ----------
    audio:
        Audio time series (values typically in [-1, 1]).
    sr:
        Sample rate in Hz.
    target_loudness:
        Desired integrated loudness in LUFS.
        Defaults to ``settings.TARGET_LOUDNESS_LUFS`` when ``None``.
    method:
        One of ``'lufs'``, ``'peak'``, or ``'rms'``.
        Defaults to ``settings.LOUDNESS_METHOD`` when ``None``.

    Returns
    -------
    np.ndarray
        Loudness-normalised audio clipped to [-1, 1] as ``float32``.

    Raises
    ------
    ValueError
        If *method* is not one of the supported strategies.
    """
    # Guard: empty or silent signal
    if len(audio) == 0 or np.max(np.abs(audio)) == 0:
        return audio

    if method is LoudnessNormalization.LUFS:
        audio_normalized = _lufs_normalize(audio, sr, target_loudness)
    elif method is LoudnessNormalization.PEAK:
        audio_normalized = _peak_normalize(audio)
    elif method == LoudnessNormalization.RMS:
        audio_normalized = _rms_normalize(audio)
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. "
            "Expected one of: 'lufs', 'peak', 'rms'."
        )

    # Hard-limit to prevent clipping
    return np.clip(audio_normalized, -1.0, 1.0).astype(np.float32)


def get_loudness(audio: np.ndarray, sr: int) -> float:
    """
    Measure the integrated loudness of *audio* in LUFS.

    Parameters
    ----------
    audio:
        Audio time series.
    sr:
        Sample rate in Hz.

    Returns
    -------
    float
        Integrated loudness (LUFS), or ``-inf`` for a silent signal.
    """
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(audio)


def is_audio_too_quiet(
        audio: np.ndarray,
        sr: int,
        threshold: float = -40.0,
) -> bool:
    """
    Return ``True`` when *audio* is too quiet for reliable ASR.

    Parameters
    ----------
    audio:
        Audio time series.
    sr:
        Sample rate in Hz.
    threshold:
        Loudness threshold in LUFS; audio below this level is flagged.

    Returns
    -------
    bool
    """
    return get_loudness(audio, sr) < threshold


def _lufs_normalize(
        audio: np.ndarray,
        sr: int,
        target_loudness: float,
) -> np.ndarray:
    """Apply ITU-R BS.1770 integrated loudness normalization."""
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)

    if np.isnan(current_loudness) or current_loudness == -np.inf:
        logger.info("Audio too quiet for LUFS measurement; falling back to peak normalization.")
        return _peak_normalize(audio)

    return pyln.normalize.loudness(audio, current_loudness, target_loudness)


def _peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Scale *audio* so that the maximum absolute sample equals *target_peak*.

    Parameters
    ----------
    audio:
        Input audio.
    target_peak:
        Desired peak amplitude (< 1.0 to provide headroom).

    Returns
    -------
    np.ndarray
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (target_peak / max_val)
    return audio


def _rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Scale *audio* so that its root-mean-square energy equals *target_rms*.

    Parameters
    ----------
    audio:
        Input audio.
    target_rms:
        Desired RMS amplitude (typical range 0.05 – 0.2).

    Returns
    -------
    np.ndarray
    """
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        return audio * (target_rms / current_rms)
    return audio
