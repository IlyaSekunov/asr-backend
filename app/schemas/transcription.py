"""Pydantic schemas for the Transcription API endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.transcribers.transcription_result import TranscriptionResult


class TaskStatus(str, Enum):
    READY = "READY"
    QUEUED = "QUEUED"
    STARTED = "STARTED"
    FAILED = "FAILED"


class TranscriptionTaskResponse(BaseModel):
    """Returned immediately after a file is uploaded and enqueued."""
    task_id: str = Field(..., description="Unique identifier of the transcription task.")


class TranscriptionTaskResultResponse(BaseModel):
    """Returned when polling task status. `result` is populated only when READY."""
    status: TaskStatus
    result: Optional[TranscriptionResult] = None


class ErrorResponse(BaseModel):
    detail: str