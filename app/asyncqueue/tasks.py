from loguru import logger

from app.pipeline.asr_pipeline_factory import create_asr_pipeline
from app.transcribers.transcription_result import TranscriptionResult
from app.util.io import load_audio, delete_file

_asr_pipeline = create_asr_pipeline()


def transcribe_task(file_path: str) -> TranscriptionResult:
    try:
        audio = load_audio(file_path)
        result = _asr_pipeline.transcribe(audio)
        return result

    finally:
        delete_file(file_path)
        logger.debug(f"Temporary file deleted: {file_path}")
