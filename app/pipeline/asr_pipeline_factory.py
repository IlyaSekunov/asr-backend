"""Factory that assembles the ASR pipeline from settings."""

from typing import List

from loguru import logger

from app.config import settings
from app.pipeline.asr_pipeline import AsrPipeline
from app.preprocessing.audio_preprocessor import AudioPreprocessor
from app.preprocessing.loudness_normalizer import LoudnessNormalizer
from app.preprocessing.noise_reducer import NoiseReducer
from app.transcribers.whisper_transcriber import WhisperTranscriber


def _configure_preprocessors() -> List[AudioPreprocessor]:
    """Return the ordered list of preprocessors based on current settings."""
    preprocessors = []

    if settings.LOUDNESS_NORMALIZATION_ENABLED:
        preprocessors.append(LoudnessNormalizer())

    if settings.DENOISING_ENABLED:
        preprocessors.append(NoiseReducer())

    return preprocessors


def _configure_postprocessors() -> List[AudioPreprocessor]:
    """Return the ordered list of postprocessors based on current settings."""
    postprocessors = []
    return postprocessors


def create_asr_pipeline() -> AsrPipeline:
    """Build and return a fully initialised ASR pipeline."""
    logger.info("Initializing ASR pipeline")

    pipeline = AsrPipeline(
        preprocessors=_configure_preprocessors(),
        postprocessors=_configure_postprocessors(),
        transcriber=WhisperTranscriber(),
    )

    logger.info("ASR pipeline initialized")
    return pipeline
