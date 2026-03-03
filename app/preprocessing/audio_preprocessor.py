"""Abstract base class for audio preprocessing steps."""

from abc import ABC, abstractmethod

import numpy as np


class AudioPreprocessor(ABC):
    """
    Interface for a single audio preprocessing step.

    Implementations must accept and return a mono float32 waveform with
    values normalised to [-1.0, 1.0]. Sample rate and array length should
    be preserved unless the transformation explicitly changes them.
    """

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Transform input audio and return the processed result.

        Parameters
        ----------
        audio : np.ndarray
            Input waveform, values in [-1.0, 1.0].

        Returns
        -------
        np.ndarray
            Processed waveform in the same format.
        """
        pass
