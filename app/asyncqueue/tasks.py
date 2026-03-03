"""
RQ task executed by the worker process.

Importing this module triggers `create_asr_pipeline()`, which loads the Whisper
model into memory. This is intentional — the model is loaded once at worker
startup and reused across all jobs. Never import this module from the API process.
"""

from loguru import logger

from app.pipeline.asr_pipeline_factory import create_asr_pipeline
from app.transcribers.transcription_result import TranscriptionResult
from app.util.io import load_audio, delete_file

# Loaded once when the worker imports this module, then reused for every job.
_asr_pipeline = create_asr_pipeline()


def transcribe_task(file_path: str) -> TranscriptionResult:
    """Load audio from disk, run the ASR pipeline, and clean up the temporary file."""
    try:
        audio = load_audio(file_path)
        return _asr_pipeline.transcribe(audio)
    finally:
        delete_file(file_path)
        logger.debug("Temporary file deleted: {}", file_path)