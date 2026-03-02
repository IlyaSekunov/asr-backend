"""
Transcription endpoint.

POST /transcribe  — accepts an audio file (.mp3 or .wav), runs the full
pre-processing → ASR pipeline, and returns a structured transcript.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from loguru import logger

from app.asyncqueue.redis_queue_manager import fetch_job_status, fetch_job_result, job_exists, \
    enqueue_transcription_task
from app.config import settings
from app.schemas.transcription import (
    ErrorResponse,
    TranscriptionTaskResponse, TranscriptionTaskResultResponse, TaskStatus,
)
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
    response_model=TranscriptionTaskResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type or oversized upload."},
        500: {"model": ErrorResponse, "description": "Internal error."},
    },
    summary="Send transcribe task to a queue",
    description=(
            "Upload a **.mp3** or **.wav** file. The server submit the transcription task "
            "to async queue and returns a task_id which can be used to get a result of transcription"
    ),
)
async def transcribe_audio(file: UploadFile) -> TranscriptionTaskResponse:
    """
    Submit the transcription of audio file (.MP3 or .WAV) to async queue.

    Parameters
    ----------
    file:
        Multipart-encoded audio file (.MP3 or .WAV).

    Returns
    -------
    TranscriptionTaskResponse
        Contains automatically generated string task_id that can be used to get a result of transcription.
    """
    _validate_file_extension(file)

    file_bytes = await file.read()
    filename = file.filename

    logger.info("Received transcription request | file={} size={}B", filename, len(file_bytes))

    file_path = save_audio_bytes(file_bytes, filename)
    task_id = enqueue_transcription_task(file_path)

    return TranscriptionTaskResponse(task_id=task_id)


@router.get(
    "/{task_id}",
    response_model=TranscriptionTaskResultResponse,
    summary="Get transcription task result",
    description=(
            "'task_id' is used for pulling the result of the transcription task. "
            "Possible task statuses: READY, QUEUED, FAILED. "
            "If status READY, additional field, which contains transcribed information, is provided."
    ),
)
async def get_transcription_result(task_id: str) -> TranscriptionTaskResultResponse:
    """
    Returns current transcription task status.

    If task is READY, returns transcription result.
    """
    if not job_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    status = fetch_job_status(task_id)
    if status != TaskStatus.READY:
        return TranscriptionTaskResultResponse(
            status=status,
            result=None,
        )

    result = fetch_job_result(task_id)
    return TranscriptionTaskResultResponse(
        status=TaskStatus.READY,
        result=result,
    )
