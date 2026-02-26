"""
Audio denoising utilities for preprocessing pipeline.

This module provides noise reduction functionality for audio signals using
the noisereduce library. It implements both stationary (constant noise profile)
and adaptive (non-stationary) noise reduction algorithms through the NoiseReducer
class, which integrates with the AudioPreprocessor inheritance hierarchy.

The module is configured via application settings and produces denoised audio
suitable for downstream tasks like transcription or feature extraction.
"""

import numpy as np
import noisereduce as nr

from app.config import settings
from app.preprocessing.audio_preprocessor import AudioPreprocessor


class NoiseReducer(AudioPreprocessor):
    """
    Audio noise reduction processor that wraps the noisereduce library.

    This preprocessor removes background noise from audio signals using
    either stationary (constant noise profile) or adaptive (non-stationary)
    noise reduction algorithms.

    Attributes
    ----------
    sr : int
        Target sample rate in Hz.
    stationary : bool
        If True, assumes constant noise profile; if False, uses adaptive model.
    prop_decrease : float
        Fraction of estimated noise energy to remove (0.0 to 1.0).
    """

    def __init__(
            self,
            sr: int = settings.TARGET_SAMPLE_RATE,
            stationary: bool = settings.DENOISE_STATIONARY,
            prop_decrease: float = settings.DENOISE_PROP_DECREASE,
    ):
        """
        Initialize the NoiseReducer with denoising parameters.

        Parameters
        ----------
        sr : int, optional
            Sample rate in Hz. Defaults to settings.TARGET_SAMPLE_RATE.
        stationary : bool, optional
            If True, use stationary noise reduction (constant noise profile).
            If False, use adaptive noise reduction. Defaults to settings.DENOISE_STATIONARY.
        prop_decrease : float, optional
            Fraction of estimated noise energy to remove, between 0.0 and 1.0.
            Higher values remove more noise but may affect signal quality.
            Defaults to settings.DENOISE_PROP_DECREASE.
        """
        self.sr = sr
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to an audio signal.

        Parameters
        ----------
        audio : np.ndarray
            Audio time series with values normalized to [-1.0, 1.0].
            Typically produced by librosa.load().

        Returns
        -------
        np.ndarray
            Denoised audio signal with the same shape and dtype as input.

        Examples
        --------
        >>> reducer = NoiseReducer(sr=16000, stationary=True)
        >>> denoised_audio = reducer.process(noisy_audio)
        """
        return nr.reduce_noise(
            y=audio,
            sr=self.sr,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
        )
