"""
Loudness normalization utilities for audio preprocessing.

This module provides loudness normalization functionality using LUFS (ITU-R BS.1770),
peak, and RMS normalization strategies. The LoudnessNormalizer class integrates
with the AudioPreprocessor hierarchy and provides methods for loudness measurement,
normalization, and silence detection.

The module supports three normalization methods:
    - LUFS: Integrated loudness normalization (perceptual)
    - PEAK: Peak amplitude normalization
    - RMS: Root-mean-square energy normalization

Typical usage:
    normalizer = LoudnessNormalizer(sr=16000, method=LoudnessNormalization.LUFS)
    normalized_audio = normalizer.process(input_audio)
    loudness = normalizer.measure_loudness(input_audio)
"""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from loguru import logger

from app.config import LoudnessNormalization, settings
from app.preprocessing.audio_preprocessor import AudioPreprocessor


class LoudnessNormalizer(AudioPreprocessor):
    """
    Audio preprocessor for loudness normalization and measurement.

    This class provides various loudness normalization strategies including
    LUFS (perceptual), peak, and RMS normalization. It integrates with the
    AudioPreprocessor pipeline and offers additional methods for loudness
    analysis and quality checks.

    The processor ensures output is properly clipped to [-1.0, 1.0] range
    and handles edge cases like silent or extremely quiet audio gracefully.

    Attributes
    ----------
    sr : int
        Target sample rate in Hz for loudness measurement.
    target_loudness : float
        Desired integrated loudness in LUFS (for LUFS method).
    method : LoudnessNormalization
        Normalization strategy to use (LUFS, PEAK, or RMS).
    target_peak : float
        Desired peak amplitude for peak normalization (< 1.0).
    target_rms : float
        Desired RMS amplitude for RMS normalization.
    """

    def __init__(
            self,
            sr: int = settings.TARGET_SAMPLE_RATE,
            target_loudness: float = settings.LOUDNESS_TARGET,
            method: LoudnessNormalization = settings.LOUDNESS_METHOD,
            target_peak: float = 0.95,
            target_rms: float = 0.1,
    ):
        """
        Initialize the LoudnessNormalizer with normalization parameters.

        Parameters
        ----------
        sr : int, optional
            Sample rate in Hz for loudness measurement.
            Defaults to settings.TARGET_SAMPLE_RATE.
        target_loudness : float, optional
            Desired integrated loudness in LUFS for LUFS normalization.
            Defaults to settings.LOUDNESS_TARGET.
        method : LoudnessNormalization, optional
            Normalization strategy to use. Must be one of:
            - LoudnessNormalization.LUFS: perceptual loudness normalization
            - LoudnessNormalization.PEAK: peak amplitude normalization
            - LoudnessNormalization.RMS: RMS energy normalization
            Defaults to settings.LOUDNESS_METHOD.
        target_peak : float, optional
            Desired peak amplitude for peak normalization, between 0.0 and 1.0.
            Values < 1.0 provide headroom to prevent clipping. Defaults to 0.95.
        target_rms : float, optional
            Desired RMS amplitude for RMS normalization, typically 0.05-0.2.
            Defaults to 0.1.
        """
        self.sr = sr
        self.target_loudness = target_loudness
        self.method = method
        self.target_peak = target_peak
        self.target_rms = target_rms

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply loudness normalization to an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Audio time series with values normalized to [-1.0, 1.0].

        Returns
        -------
        np.ndarray
            Loudness-normalized audio clipped to [-1.0, 1.0] as float32.

        Raises
        ------
        ValueError
            If the normalization method is not supported.

        Examples
        --------
        >>> normalizer = LoudnessNormalizer(sr=16000, method=LoudnessNormalization.LUFS)
        >>> normalized = normalizer.process(input_audio)
        """
        # Guard: empty or silent signal
        if len(audio) == 0 or np.max(np.abs(audio)) == 0:
            logger.debug("Empty or silent audio detected; returning unchanged.")
            return audio

        # Apply selected normalization method
        if self.method == LoudnessNormalization.LUFS:
            audio_normalized = self._lufs_normalize(audio)
        elif self.method == LoudnessNormalization.PEAK:
            audio_normalized = self._peak_normalize(audio)
        elif self.method == LoudnessNormalization.RMS:
            audio_normalized = self._rms_normalize(audio)
        else:
            raise ValueError(
                f"Unknown normalization method: '{self.method}'. "
                f"Expected one of: {[m.value for m in LoudnessNormalization]}"
            )

        # Hard-limit to prevent clipping
        return np.clip(audio_normalized, -1.0, 1.0).astype(np.float32)

    def measure_loudness(self, audio: np.ndarray) -> float:
        """
        Measure the integrated loudness of audio in LUFS.

        Parameters
        ----------
        audio : np.ndarray
            Audio time series to measure.

        Returns
        -------
        float
            Integrated loudness in LUFS, or -inf for silent signals.

        Examples
        --------
        >>> normalizer = LoudnessNormalizer(sr=16000)
        >>> loudness = normalizer.measure_loudness(audio)
        >>> print(f"Current loudness: {loudness:.1f} LUFS")
        """
        meter = pyln.Meter(self.sr)
        return meter.integrated_loudness(audio)

    def _lufs_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply ITU-R BS.1770 integrated loudness normalization.

        Falls back to peak normalization if audio is too quiet for reliable
        LUFS measurement.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal.

        Returns
        -------
        np.ndarray
            LUFS-normalized audio.
        """
        meter = pyln.Meter(self.sr)
        current_loudness = meter.integrated_loudness(audio)

        if np.isnan(current_loudness) or current_loudness == -np.inf:
            logger.info(
                f"Audio too quiet for LUFS measurement (loudness: {current_loudness}); "
                "falling back to peak normalization."
            )
            return self._peak_normalize(audio)

        logger.debug(f"Normalizing LUFS from {current_loudness:.1f} to {self.target_loudness:.1f}")
        return pyln.normalize.loudness(audio, current_loudness, self.target_loudness)

    def _peak_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Scale audio so that maximum absolute sample equals target_peak.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal.

        Returns
        -------
        np.ndarray
            Peak-normalized audio.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            gain = self.target_peak / max_val
            logger.debug(f"Peak normalizing with gain: {gain:.3f}")
            return audio * gain
        return audio

    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Scale audio so that root-mean-square energy equals target_rms.

        Parameters
        ----------
        audio : np.ndarray
            Input audio signal.

        Returns
        -------
        np.ndarray
            RMS-normalized audio.
        """
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            gain = self.target_rms / current_rms
            logger.debug(f"RMS normalizing from {current_rms:.4f} to {self.target_rms:.4f} (gain: {gain:.3f})")
            return audio * gain
        return audio
