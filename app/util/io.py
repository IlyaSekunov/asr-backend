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
    Stream an uploaded audio file to disk without loading it fully into memory.

    Writes data in 512 KB chunks to a temporary file, then atomically renames
    it to the final path. This ensures the worker never reads a partially
    written file — it only sees the file after the rename completes.

    Parameters
    ----------
    file : UploadFile
        Incoming multipart file from FastAPI.
    task_id : str
        Unique task identifier used as part of the final filename
        to avoid collisions between concurrent uploads.

    Returns
    -------
    str
        Absolute path to the saved audio file.

    Raises
    ------
    IOError
        If writing fails for any reason. The temporary file is cleaned
        up automatically before the exception is re-raised.
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

        # Atomic rename: the final path becomes visible to the worker
        # only after the file is fully written.
        os.rename(tmp_path, final_path)
        return str(final_path.absolute())

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise IOError(f"Failed to write audio file: {e}")


def load_audio(file_path: str) -> np.ndarray:
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
    audio, _ = librosa.load(file_path, sr=settings.TARGET_SAMPLE_RATE)
    return audio


def delete_file(file_path: str) -> None:
    """
    Safely delete a temporary audio file, ignoring if it doesn't exist.

    Parameters
    ----------
    file_path : str
        Path to the file to delete.
    """
    try:
        os.unlink(file_path)
    except FileNotFoundError:
        # File already deleted or never created - this is fine
        pass
    except PermissionError as e:
        logger.warning(f"Cannot delete file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error deleting {file_path}: {e}")
