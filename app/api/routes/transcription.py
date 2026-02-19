"""
Transcription endpoint.

POST /transcribe  — accepts an audio file (.mp3 or .wav), runs the full
pre-processing → ASR pipeline, and returns a structured transcript.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from loguru import logger

from app.core.config import settings
from app.schemas.transcription import (
    ErrorResponse,
    TranscriptionResponse,
)
from app.services.asr import transcribe
from app.services.audio import load_audio, run_preprocessing_pipeline
from app.util.io import save_audio_bytes

router = APIRouter(prefix="/transcribe", tags=["transcription"])


def _validate_file_extension(file: UploadFile) -> None:
    """
    Raise a 400 if the uploaded file has an unsupported extension.

    Parameters
    ----------
    file:
        The ``UploadFile`` object received from FastAPI.

    Raises
    ------
    HTTPException
        400 Bad Request when the extension is not in the allowed set.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed types: {sorted(settings.ALLOWED_AUDIO_EXTENSIONS)}"
            ),
        )


@router.post(
    "/",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or oversized upload."},
        500: {"model": ErrorResponse, "description": "Internal processing error."},
    },
    summary="Transcribe an audio file",
    description=(
            "Upload a **.mp3** or **.wav** file. The server applies configurable "
            "pre-processing (noise reduction, loudness normalization) and returns "
            "a full transcript with per-segment timestamps."
    ),
)
async def transcribe_audio(file: UploadFile) -> TranscriptionResponse:
    """
    Transcribe the uploaded audio file.

    The endpoint is intentionally *synchronous* with respect to the ML work so
    that it can be trivially wrapped in a Celery task later.  All heavy I/O
    and compute happens inside synchronous functions; FastAPI's thread-pool
    executor prevents the event loop from blocking.

    Parameters
    ----------
    file:
        Multipart-encoded audio file.

    Returns
    -------
    TranscriptionResponse
        Structured transcription with segments, language, and metadata.
    """
    _validate_file_extension(file)

    file_bytes = await file.read()
    filename = file.filename or "upload"

    logger.info("Received transcription request | file={} size={}B", filename, len(file_bytes))

    try:
        file_path = save_audio_bytes(file_bytes, filename)
        audio, sr = load_audio(file_path)
        audio = run_preprocessing_pipeline(audio, sr)
        result = transcribe(audio)
    except Exception as exc:
        logger.exception("Transcription pipeline failed | file={}", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {exc}",
        ) from exc

    return TranscriptionResponse(
        filename=filename,
        language=result.language,
        language_probability=result.language_probability,
        full_text=result.text,
    )
