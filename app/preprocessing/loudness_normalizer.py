"""Loudness normalisation preprocessor (LUFS, Peak, RMS)."""

from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from loguru import logger

from app.config import LoudnessNormalization, settings
from app.preprocessing.audio_preprocessor import AudioPreprocessor


class LoudnessNormalizer(AudioPreprocessor):
    """
    Normalises audio loudness using one of three strategies:
      - LUFS: perceptual integrated loudness (ITU-R BS.1770), the default.
      - PEAK: scales so the loudest sample reaches `target_peak`.
      - RMS:  scales so the RMS energy equals `target_rms`.

    Output is hard-clipped to [-1.0, 1.0] as float32.
    """

    def __init__(
            self,
            sr: int = settings.TARGET_SAMPLE_RATE,
            target_loudness: float = settings.LOUDNESS_TARGET,
            method: LoudnessNormalization = settings.LOUDNESS_METHOD,
            target_peak: float = 0.95,
            target_rms: float = 0.1,
    ):
        self.sr = sr
        self.target_loudness = target_loudness
        self.method = method
        self.target_peak = target_peak
        self.target_rms = target_rms

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Normalise loudness and clip output to [-1.0, 1.0]."""
        if len(audio) == 0 or np.max(np.abs(audio)) == 0:
            logger.debug("Empty or silent audio — skipping normalisation.")
            return audio

        if self.method == LoudnessNormalization.LUFS:
            normalised = self._lufs_normalize(audio)
        elif self.method == LoudnessNormalization.PEAK:
            normalised = self._peak_normalize(audio)
        elif self.method == LoudnessNormalization.RMS:
            normalised = self._rms_normalize(audio)
        else:
            raise ValueError(
                f"Unknown normalisation method: '{self.method}'. "
                f"Expected one of: {[m.value for m in LoudnessNormalization]}"
            )

        return np.clip(normalised, -1.0, 1.0).astype(np.float32)

    def measure_loudness(self, audio: np.ndarray) -> float:
        """Return integrated loudness in LUFS, or -inf for silent audio."""
        meter = pyln.Meter(self.sr)
        return meter.integrated_loudness(audio)

    def _lufs_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply ITU-R BS.1770 loudness normalisation.

        Falls back to peak normalisation when the signal is too quiet for a
        reliable LUFS measurement (returns -inf or NaN).
        """
        meter = pyln.Meter(self.sr)
        current_loudness = meter.integrated_loudness(audio)

        if np.isnan(current_loudness) or current_loudness == -np.inf:
            # Fallback is a deviation from expected behaviour — warn, not info.
            logger.warning(
                "Audio too quiet for LUFS measurement ({}); falling back to peak normalisation.",
                current_loudness,
            )
            return self._peak_normalize(audio)

        logger.debug("LUFS normalising {} → {} LUFS", current_loudness, self.target_loudness)
        return pyln.normalize.loudness(audio, current_loudness, self.target_loudness)

    def _peak_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Scale so the loudest sample equals `target_peak`."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            gain = self.target_peak / max_val
            logger.debug("Peak normalising with gain {:.3f}", gain)
            return audio * gain
        return audio

    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """Scale so RMS energy equals `target_rms`."""
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            gain = self.target_rms / current_rms
            logger.debug("RMS normalising {:.4f} → {:.4f} (gain {:.3f})", current_rms, self.target_rms, gain)
            return audio * gain
        return audio
