"""Audio upload helper — used by the API process only."""

import os
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile, status

from app.config import settings

# Derived once at import time so the arithmetic isn't repeated on every request.
_MAX_FILE_SIZE_BYTES: int = settings.MAX_AUDIO_FILE_SIZE_MB * 1024 * 1024
_CHUNK_SIZE: int = 512 * 1024  # 512 KB


def validate_file_size_header(file: UploadFile) -> None:
    """
    Reject the request early when the client supplies a Content-Length header
    that already exceeds the limit.

    This is a *fast-path* check only — Content-Length is optional and
    untrustworthy, so the hard limit is enforced again during streaming in
    :func:`save_audio_stream`.

    Raises
    ------
    HTTPException
        HTTP 413 if the declared size exceeds :data:`_MAX_FILE_SIZE_BYTES`.
    """
    if file.size is not None and file.size > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File too large: {file.size / (1024 * 1024):.1f} MB received, "
                f"maximum allowed is {settings.MAX_AUDIO_FILE_SIZE_MB} MB."
            ),
        )


async def save_audio_stream(file: UploadFile, task_id: str) -> str:
    """
    Stream an uploaded audio file to disk in 512 KB chunks, enforcing a hard
    size limit regardless of whether Content-Length was provided by the client.

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
    HTTPException
        HTTP 413 if the total bytes written exceeds the configured limit.
    IOError
        If writing fails for any other reason. The temporary file is cleaned
        up before re-raising.
    """
    output_path = Path(settings.AUDIO_STORAGE_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    final_path = output_path / f"{safe_name}_{task_id}"
    tmp_path = final_path.with_suffix(".tmp")

    bytes_written = 0

    try:
        async with aiofiles.open(tmp_path, "wb") as f:
            while chunk := await file.read(_CHUNK_SIZE):
                bytes_written += len(chunk)

                if bytes_written > _MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File too large: exceeds the maximum allowed size of "
                            f"{settings.MAX_AUDIO_FILE_SIZE_MB} MB."
                        ),
                    )

                await f.write(chunk)

        os.rename(tmp_path, final_path)
        return str(final_path.absolute())

    except HTTPException:
        # Clean up the partial file before propagating the 413.
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise IOError(f"Failed to write audio file: {e}")
