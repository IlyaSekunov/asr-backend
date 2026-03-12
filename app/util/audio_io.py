"""Audio file I/O helpers — used by the worker process only."""

import os

import librosa
import numpy as np
from loguru import logger

from app.config import settings


def load_audio(file_path: str) -> np.ndarray:
    """
    Load an audio file as a mono float32 array resampled to TARGET_SAMPLE_RATE.

    Parameters
    ----------
    file_path : str
        Path to a .mp3 or .wav file.

    Returns
    -------
    np.ndarray
        Mono float32 waveform.
    """
    audio, _ = librosa.load(file_path, sr=settings.TARGET_SAMPLE_RATE)
    return audio


def delete_file(file_path: str) -> None:
    """
    Delete a file, ignoring FileNotFoundError.

    Logs a warning on PermissionError and an error for any other unexpected
    exception, but never raises — a cleanup failure should not affect the
    task result already stored in Redis.
    """
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass
    except PermissionError as e:
        logger.warning("Cannot delete file {}: {}", file_path, e)
    except Exception as e:
        logger.error("Unexpected error deleting {}: {}", file_path, e)
