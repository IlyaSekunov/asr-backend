"""
Pydantic schemas for the Transcription API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    filename: str = Field(..., description="Original uploaded filename.")
    language: str = Field(..., description="Detected language code (e.g. 'en').")
    language_probability: float = Field(..., description="Confidence of language detection (0–1).")
    full_text: str = Field(..., description="Complete transcript as a single string.")


class ErrorResponse(BaseModel):
    detail: str


# Re-export for convenience
TranscriptionResponse.model_rebuild()
