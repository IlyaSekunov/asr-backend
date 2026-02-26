"""
ASR Pipeline that chains audio preprocessing with speech transcription.

This module provides a pipeline abstraction that combines multiple audio
preprocessing steps with a speech recognition model. The pipeline ensures
consistent processing flow and proper integration between preprocessing
and transcription stages.
"""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger

from app.transcribers.audio_transcriber import AudioTranscriber
from app.transcribers.transcription_result import TranscriptionResult
from app.transcribers.whisper_transcriber import WhisperTranscriber
from app.config import settings
from app.preprocessing.audio_preprocessor import AudioPreprocessor
from app.preprocessing.loudness_normalizer import LoudnessNormalizer
from app.preprocessing.noise_reducer import NoiseReducer


class AsrPipeline:
    """
    Pipeline that sequentially applies audio preprocessing followed by transcription.

    This class orchestrates the complete ASR workflow by:
        1. Applying a series of audio preprocessing steps in order
        2. Passing the processed audio to a transcriber
        3. Returning the final transcription result

    The pipeline is designed to be flexible and composable, allowing different
    combinations of preprocessors and transcribers while maintaining a simple
    interface for the caller.

    Attributes
    ----------
    preprocessors : List[AudioPreprocessor]
        Ordered list of audio preprocessing steps to apply sequentially.
        Each preprocessor must implement the AudioPreprocessor interface.
        Applied in the order they appear in the list.

    transcriber : AudioTranscriber
        The speech recognition model that converts processed audio to text.
        Must implement the AudioTranscriber interface.

    Notes
    -----
    - Preprocessors are applied in the order they are provided
    - Each preprocessor receives the output of the previous one
    - The pipeline logs each step for debugging and monitoring
    - Empty preprocessor list is allowed (direct transcription)
    - The pipeline does not modify the original input audio

    Examples
    --------
    >>> # Basic pipeline with single preprocessor
    >>> pipeline = AsrPipeline(
    ...     preprocessors=[NoiseReducer()],
    ...     transcriber=WhisperTranscriber()
    ... )
    >>> result = pipeline.process(audio)

    >>> # Complex pipeline with multiple preprocessing steps
    >>> pipeline = AsrPipeline(
    ...     preprocessors=[
    ...         NoiseReducer(stationary=True),
    ...         LoudnessNormalizer(target_loudness=-23.0),
    ...     ],
    ...     transcriber=WhisperTranscriber()
    ... )

    >>> # Minimal pipeline (direct transcription)
    >>> pipeline = AsrPipeline(
    ...     preprocessors=[],
    ...     transcriber=WhisperTranscriber()
    ... )
    """

    def __init__(
            self,
            preprocessors: List[AudioPreprocessor],
            transcriber: AudioTranscriber,
    ):
        """
        Initialize the ASR pipeline with preprocessing steps and transcriber.

        Parameters
        ----------
        preprocessors : List[AudioPreprocessor]
            Ordered list of audio preprocessing steps. Can be empty if no
            preprocessing is required. Each element must be an instance of
            a class implementing the AudioPreprocessor interface.

        transcriber : AudioTranscriber
            The speech recognition model to use for transcription. Must be
            an instance of a class implementing the AudioTranscriber interface.
        """
        self.preprocessors = preprocessors
        self.transcriber = transcriber

        logger.debug(
            f"Initialized AsrPipeline with {len(preprocessors)} preprocessors "
            f"and {type(transcriber).__name__}"
        )

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Execute the complete ASR pipeline on input audio.

        This method sequentially:
            1. Applies all preprocessing steps to the input audio
            2. Passes the processed audio to the transcriber
            3. Returns the structured transcription result

        Each preprocessing step receives the output of the previous step,
        allowing for complex audio transformation chains. The pipeline logs
        progress at each stage for monitoring and debugging.

        Parameters
        ----------
        audio : np.ndarray
            Input audio time series. Expected format:
            - Shape can be (n_samples,) for mono or (n_channels, n_samples)
              for multi-channel audio
            - dtype should be float32 or float64
            - Sample rate must be compatible with first preprocessor

        Returns
        -------
        TranscriptionResult
            Structured transcription result containing:
            - Transcribed text from the audio
            - Detected language (ISO 639-1 code)
            - Language detection confidence (0.0 to 1.0)

        Raises
        ------
        ValueError
            If the input audio is invalid (empty array, wrong dimensions, etc.)
        RuntimeError
            If any preprocessing step fails or transcription fails

        Notes
        -----
        - The pipeline does not cache intermediate results
        - Each call to process() is independent
        - The original input audio is never modified
        """
        logger.info(f"Starting ASR pipeline")

        # Validate input
        if len(audio) == 0:
            raise ValueError("Input audio cannot be empty")

        if not np.isfinite(audio).all():
            raise ValueError("Input audio contains infinite or NaN values")

        current_audio = audio

        # Apply preprocessing steps sequentially
        for preprocessor in self.preprocessors:
            current_audio = preprocessor.process(current_audio)

        result = self.transcriber.transcribe(current_audio)

        # Log completion
        logger.info(
            f"ASR pipeline complete | language={result.language} "
            f"confidence={result.language_probability:.2%} "
            f"text_preview=\"{result.text[:100]}...\""
        )

        return result


def _configure_preprocessors() -> List[AudioPreprocessor]:
    preprocessors = []
    if settings.LOUDNESS_NORMALIZATION_ENABLED:
        preprocessors.append(LoudnessNormalizer())

    if settings.DENOISING_ENABLED:
        preprocessors.append(NoiseReducer())

    return preprocessors


asr_pipeline = AsrPipeline(
    preprocessors=_configure_preprocessors(),
    transcriber=WhisperTranscriber()
)
