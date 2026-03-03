"""File I/O helpers for audio upload and loading."""

import os
from pathlib import Path

import aiofiles
import librosa
import numpy as np
from fastapi import UploadFile
from loguru import logger

from app.config import settings


async def save_audio_stream(file: UploadFile, task_id: str) -> str:
    """
    Stream an uploaded audio file to disk in 512 KB chunks.

    Writes to a `.tmp` file first, then atomically renames it so the worker
    never reads a partially written file.

    Parameters
    ----------
    file : UploadFile
        Incoming multipart file from FastAPI.
    task_id : str
        Appended to the filename to prevent collisions between concurrent uploads.

    Returns
    -------
    str
        Absolute path to the saved audio file.

    Raises
    ------
    IOError
        If writing fails. The temporary file is cleaned up before re-raising.
    """
    output_path = Path(settings.AUDIO_STORAGE_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    final_path = output_path / f"{safe_name}_{task_id}"
    tmp_path = final_path.with_suffix(".tmp")

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            while chunk := await file.read(512 * 1024):
                await f.write(chunk)
        os.rename(tmp_path, final_path)
        return str(final_path.absolute())

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise IOError(f"Failed to write audio file: {e}")


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
    """Delete a file, ignoring FileNotFoundError."""
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        pass
    except PermissionError as e:
        logger.warning("Cannot delete file {}: {}", file_path, e)
    except Exception as e:
        logger.error("Unexpected error deleting {}: {}", file_path, e)
