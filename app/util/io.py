import os
import uuid
from pathlib import Path

import librosa
import numpy as np
from loguru import logger

from app.config import settings


def save_audio_bytes(audio_file: bytes, original_filename: str) -> str:
    """
    Save audio bytes to a temporary file on disk.

    This function creates a unique filename, ensures the output directory exists,
    and writes the audio bytes to a file. Returns the path to the saved file.

    Args:
        audio_file: Raw audio data as bytes (e.g., from an uploaded MP3 or WAV file)
        original_filename: Name of uploaded audio file

    Returns:
        str: Absolute path to the saved file

    Raises:
        ValueError: If audio_file is empty
        IOError: If file writing fails
    """
    if not audio_file:
        raise ValueError("Audio file bytes cannot be empty")

    # Create output directory if it doesn't exist
    output_path = Path(settings.AUDIO_STORAGE_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate unique filename to avoid collisions
    # Using UUID ensures uniqueness even with concurrent requests
    filename = f"{original_filename}_{uuid.uuid4()}"
    file_path = output_path / filename

    try:
        # Write bytes to file
        with open(file_path, 'wb') as f:
            f.write(audio_file)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk (optional, for extra safety)

        return str(file_path.absolute())

    except IOError as e:
        # Clean up if file was partially created
        if file_path.exists():
            file_path.unlink()
        raise IOError(f"Failed to write audio file: {e}")


def extract_filename(file_path: str, include_extension: bool = True) -> str:
    """
    Extract filename from an absolute file path.

    This function takes a full file path and returns just the filename.
    It works cross-platform (Windows, Linux, macOS) and handles various edge cases.

    Args:
        file_path: Absolute or relative path to a file
        include_extension: If True, returns filename with extension.
                          If False, returns filename without extension.

    Returns:
        str: Extracted filename

    Raises:
        ValueError: If the path is empty or invalid
    """
    # Validate input
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")

    filename = os.path.basename(file_path)

    if not include_extension:
        filename = os.path.splitext(filename)[0]

    return filename


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
