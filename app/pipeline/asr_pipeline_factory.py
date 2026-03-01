from typing import List

from app.config import settings
from app.pipeline.asr_pipeline import AsrPipeline
from app.preprocessing.audio_preprocessor import AudioPreprocessor
from app.preprocessing.loudness_normalizer import LoudnessNormalizer
from app.preprocessing.noise_reducer import NoiseReducer
from app.transcribers.whisper_transcriber import WhisperTranscriber
from loguru import logger


def _configure_preprocessors() -> List[AudioPreprocessor]:
    preprocessors = []
    if settings.LOUDNESS_NORMALIZATION_ENABLED:
        preprocessors.append(LoudnessNormalizer())

    if settings.DENOISING_ENABLED:
        preprocessors.append(NoiseReducer())

    return preprocessors


def create_asr_pipeline() -> AsrPipeline:
    logger.info("Initializing ASR pipeline")

    pipeline = AsrPipeline(
        preprocessors=_configure_preprocessors(),
        transcriber=WhisperTranscriber(),
    )

    logger.info("ASR pipeline initializing has finished")
    return pipeline
