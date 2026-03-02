"""
Pydantic schemas for the Transcription API endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.transcribers.transcription_result import TranscriptionResult


class TaskStatus(str, Enum):
    READY = "READY"
    QUEUED = "QUEUED"
    FAILED = "FAILED"
    STARTED = "STARTED"


class TranscriptionTaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique identifier of the created transcription task.")


class TranscriptionTaskResultResponse(BaseModel):
    status: TaskStatus = Field(..., description="Current execution status of the transcription task.")
    result: Optional[TranscriptionResult] = Field(
        None,
        description="Transcription result. Present only when status is READY.",
    )


class TranscriptionResponse(BaseModel):
    filename: str = Field(..., description="Original uploaded filename.")
    language: str = Field(..., description="Detected language code (e.g. 'en').")
    language_probability: float = Field(..., description="Confidence of language detection (0–1).")
    full_text: str = Field(..., description="Complete transcript as a single string.")


class ErrorResponse(BaseModel):
    detail: str


# Re-export for convenience
TranscriptionResponse.model_rebuild()
