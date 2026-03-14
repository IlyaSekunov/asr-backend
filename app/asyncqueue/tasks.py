"""
RQ task executed by the worker process.

Importing this module triggers ``create_asr_pipeline()``, which loads the
Whisper model into memory.  This is intentional — the model is loaded once
at worker startup and reused across all jobs.  Never import this module
from the API process.
"""

import time

from loguru import logger

from app.config import settings
from app.metrics.metrics import (
    audio_duration_minutes,
    language_detections_total,
    real_time_factor,
    transcription_failure_total,
    transcription_success_total,
)
from app.pipeline.asr_pipeline_factory import create_asr_pipeline
from app.transcribers.transcription_result import TranscriptionResult
from app.util.audio_io import delete_file, load_audio

# Loaded once when the worker imports this module, then reused for every job.
_asr_pipeline = create_asr_pipeline()


def transcribe_task(file_path: str) -> TranscriptionResult:
    """Load audio from disk, run the ASR pipeline, and clean up the temp file."""
    try:
        audio = load_audio(file_path)

        duration_seconds = len(audio) / settings.TARGET_SAMPLE_RATE
        duration_minutes = duration_seconds / 60.0
        audio_duration_minutes.observe(duration_minutes)

        t0 = time.perf_counter()
        result = _asr_pipeline.transcribe(audio)
        elapsed_seconds = time.perf_counter() - t0

        # RTF = processing_time / audio_duration  (< 1.0 → faster than real-time)
        if duration_seconds > 0:
            rtf = elapsed_seconds / duration_seconds
            real_time_factor.observe(rtf)
            logger.debug(
                "RTF={:.3f} | elapsed={:.2f}s audio={:.2f}s",
                rtf,
                elapsed_seconds,
                duration_seconds,
            )

        transcription_success_total.inc()
        language_detections_total.labels(language=result.language).inc()

        return result

    except Exception:
        transcription_failure_total.inc()
        raise

    finally:
        delete_file(file_path)
        logger.debug("Temporary file deleted: {}", file_path)
