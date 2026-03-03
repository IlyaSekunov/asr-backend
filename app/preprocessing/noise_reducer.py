"""Noise reduction preprocessor backed by the noisereduce library."""

import numpy as np
import noisereduce as nr

from app.config import settings
from app.preprocessing.audio_preprocessor import AudioPreprocessor


class NoiseReducer(AudioPreprocessor):
    """
    Removes background noise using stationary or adaptive noise reduction.

      - Stationary:  assumes a constant noise profile throughout the clip.
      - Adaptive:    re-estimates the noise profile over time; better for
                     recordings where background noise changes (e.g. wind, crowd).
    """

    def __init__(
            self,
            sr: int = settings.TARGET_SAMPLE_RATE,
            stationary: bool = settings.DENOISE_STATIONARY,
            prop_decrease: float = settings.DENOISE_PROP_DECREASE,
    ):
        """
        Parameters
        ----------
        sr : int
            Sample rate of the input audio.
        stationary : bool
            True for constant noise profile; False for adaptive estimation.
        prop_decrease : float
            Fraction of noise energy to remove [0.0, 1.0]. Higher values
            remove more noise but risk introducing artefacts.
        """
        self.sr = sr
        self.stationary = stationary
        self.prop_decrease = prop_decrease

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction and return the denoised waveform."""
        return nr.reduce_noise(
            y=audio,
            sr=self.sr,
            stationary=self.stationary,
            prop_decrease=self.prop_decrease,
        )
