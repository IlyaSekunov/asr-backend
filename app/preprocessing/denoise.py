"""
Audio denoising utilities.

Provides a stationary / adaptive noise-reduction function built on top of
the ``noisereduce`` library.
"""

import numpy as np
import noisereduce as nr

from app.core.config import settings


def denoise_audio(
        audio: np.ndarray,
        sr: int,
        stationary: bool = settings.DENOISE_STATIONARY,
        prop_decrease: float = settings.DENOISE_PROP_DECREASE,
) -> np.ndarray:
    """
    Reduce background noise from an audio signal.

    Parameters
    ----------
    audio:
        Audio time series produced by ``librosa.load`` (values in [-1, 1]).
    sr:
        Sample rate of *audio* in Hz.
    stationary:
        When ``True`` (default) the noise profile is assumed to be constant.
        Set to ``False`` to use the adaptive (non-stationary) model.
        Falls back to ``settings.DENOISE_STATIONARY`` when ``None``.
    prop_decrease:
        Fraction of estimated noise energy to remove (0.0 – 1.0).
        Falls back to ``settings.DENOISE_PROP_DECREASE`` when ``None``.

    Returns
    -------
    np.ndarray
        Denoised audio with the same shape and dtype as *audio*.
    """
    return nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
    )
