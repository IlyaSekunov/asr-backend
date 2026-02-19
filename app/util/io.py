import os
import uuid
from pathlib import Path
from typing import Union


def save_audio_bytes(audio_file: bytes, output_dir: Union[str, Path] = "temp_audio") -> str:
    """
    Save audio bytes to a temporary file on disk.

    This function creates a unique filename, ensures the output directory exists,
    and writes the audio bytes to a file. Returns the path to the saved file.

    Args:
        audio_file: Raw audio data as bytes (e.g., from an uploaded MP3 or WAV file)
        output_dir: Directory where the file will be saved (default: "temp_audio")

    Returns:
        str: Absolute path to the saved file

    Raises:
        ValueError: If audio_file is empty
        IOError: If file writing fails
    """
    if not audio_file:
        raise ValueError("Audio file bytes cannot be empty")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate unique filename to avoid collisions
    # Using UUID ensures uniqueness even with concurrent requests
    filename = f"{uuid.uuid4()}.audio"
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
        filename = os.path.splitext(filename_with_ext)[0]

    return filename
