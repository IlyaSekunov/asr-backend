"""
Transcription endpoints.

POST /transcribe/      — upload audio, enqueue transcription, return task_id.
GET  /transcribe/{id}  — poll task status; result included when READY.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status
from loguru import logger

from app.asyncqueue.redis_queue_manager import (
    delete_job,
    enqueue_transcription_task,
    fetch_job_result,
    fetch_job_status,
    job_exists,
)
from app.config import settings
from app.schemas.transcription import (
    ErrorResponse,
    TaskStatus,
    TranscriptionTaskResponse,
    TranscriptionTaskResultResponse,
)
from app.util.audio_upload import save_audio_stream
from app.util.tasks import generate_task_id

router = APIRouter(prefix="/transcribe", tags=["transcription"])


def _validate_file_extension(file: UploadFile) -> None:
    """Raise HTTP 400 if the uploaded file's extension is not supported."""
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
    """Stream the uploaded file to disk and enqueue a transcription job."""
    _validate_file_extension(file)

    if not file.size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file cannot be empty",
        )

    logger.info("Received transcription request | file={} size={}", file.filename, file.size)

    task_id = generate_task_id()
    try:
        file_path = await save_audio_stream(file, task_id)
    except IOError as e:
        logger.error("Failed to save audio file | file={} task_id={} error={}", file.filename, task_id, e)
        raise HTTPException(status_code=500, detail=str(e))

    enqueue_transcription_task(file_path, task_id)
    logger.info("Task enqueued | task_id={} file={}", task_id, file.filename)

    return TranscriptionTaskResponse(task_id=task_id)


@router.get(
    "/{task_id}",
    response_model=TranscriptionTaskResultResponse,
    summary="Get transcription task result",
    description=(
            "Poll by `task_id`. Possible statuses: QUEUED, STARTED, READY, FAILED. "
            "The `result` field is populated only when status is READY."
    ),
)
async def get_transcription_result(task_id: str) -> TranscriptionTaskResultResponse:
    """Return the current status of a transcription task, and its result if ready."""
    if not job_exists(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    job_status = fetch_job_status(task_id)
    logger.debug("Task polled | task_id={} status={}", task_id, job_status)

    if job_status == TaskStatus.FAILED:
        logger.warning("Task failed | task_id={}", task_id)
        return TranscriptionTaskResultResponse(status=job_status, result=None)

    if job_status != TaskStatus.READY:
        return TranscriptionTaskResultResponse(status=job_status, result=None)

    result = fetch_job_result(task_id)
    # Clean up immediately after fetch; TTL is a safety net, not the primary mechanism.
    delete_job(task_id)
    logger.info(
        "Task result fetched and cleaned up | task_id={} language={} confidence={:.2%}",
        task_id,
        result.language,
        result.language_probability,
    )

    return TranscriptionTaskResultResponse(status=TaskStatus.READY, result=result)
