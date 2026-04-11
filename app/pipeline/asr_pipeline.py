"""ASR pipeline: chains audio preprocessors then runs transcription."""

from __future__ import annotations

from typing import List

import numpy as np
from loguru import logger

from app.postprocessing.text_postprocessor import TextPostprocessor
from app.preprocessing.audio_preprocessor import AudioPreprocessor
from app.transcribers.audio_transcriber import AudioTranscriber
from app.transcribers.transcription_result import TranscriptionResult


class AsrPipeline:
    """
    Runs a sequence of audio preprocessors followed by a transcriber.

    Preprocessors are applied in list order; each receives the output of the
    previous one. An empty preprocessor list is valid (direct transcription).
    """

    def __init__(
            self,
            preprocessors: List[AudioPreprocessor],
            postprocessors: List[TextPostprocessor],
            transcriber: AudioTranscriber,
    ):
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors
        self.transcriber = transcriber

        logger.debug(
            "Initialized AsrPipeline | preprocessors={} | postprocessors={} | transcriber={}",
            len(preprocessors),
            len(postprocessors),
            type(transcriber).__name__,
        )

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Apply preprocessing steps then transcribe.

        Parameters
        ----------
        audio : np.ndarray
            Raw mono waveform compatible with the first preprocessor.

        Returns
        -------
        TranscriptionResult
            Transcribed text, language code, and language confidence.

        Raises
        ------
        ValueError
            If the audio is empty or contains non-finite values.
        """
        if len(audio) == 0:
            raise ValueError("Input audio cannot be empty")
        if not np.isfinite(audio).all():
            raise ValueError("Input audio contains infinite or NaN values")

        current_audio = audio
        for preprocessor in self.preprocessors:
            current_audio = preprocessor.process(current_audio)

        result = self.transcriber.transcribe(current_audio)

        for postprocessor in self.postprocessors:
            result = postprocessor.process(result)

        # Only append "..." when the text was actually truncated.
        preview = result.text[:100] + ("..." if len(result.text) > 100 else "")
        logger.info(
            "ASR pipeline complete | language={} confidence={:.2%} preview=\"{}\"",
            result.language,
            result.language_probability,
            preview,
        )

        return result
