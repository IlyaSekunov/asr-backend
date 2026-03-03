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
from app.util.io import save_audio_stream
from app.util.tasks import generate_task_id

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
        400: {"model": ErrorResponse, "description": "Invalid file type or empty file."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
    summary="Submit an audio file for transcription",
    description=(
        "Upload a **.mp3** or **.wav** file. The file is streamed to disk and a "
        "transcription task is placed on the async queue. Returns a `task_id` "
        "that can be used to poll for the result."
    ),
)
async def transcribe_audio(file: UploadFile) -> TranscriptionTaskResponse:
    """
    Accept an audio file upload and enqueue it for transcription.

    The file is streamed to disk in chunks to avoid loading it fully into
    memory. The transcription itself is handled asynchronously by a worker —
    use GET /transcribe/{task_id} to poll for the result.

    Parameters
    ----------
    file : UploadFile
        Multipart-encoded audio file (.mp3 or .wav).

    Returns
    -------
    TranscriptionTaskResponse
        Contains the `task_id` assigned to this transcription job.

    Raises
    ------
    HTTPException
        400 if the file type is unsupported or the file is empty.
        500 if saving the file to disk fails.
    """
    _validate_file_extension(file)

    if not file.size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file cannot be empty"
        )

    logger.info("Received transcription request | file={}", file.filename)

    task_id = generate_task_id()
    try:
        file_path = await save_audio_stream(file, task_id)
    except IOError as e:
        raise HTTPException(status_code=500, detail=str(e))

    enqueue_transcription_task(file_path, task_id)
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
