"""Abstract base class for transcription post-processing steps."""

from abc import ABC, abstractmethod

from app.transcribers.transcription_result import TranscriptionResult


class TextPostprocessor(ABC):
    """
    Interface for a single transcription post-processing step.

    Implementations receive a :class:`TranscriptionResult`
    and must return a new instance with the processed text.
    Metadata fields — ``language`` and ``language_probability`` — should be
    preserved unless the transformation explicitly changes them (e.g. a
    language-override step).

    Post-processors are designed to be chained: each receives the output of
    the previous one, so they must be composable and side-effect-free.
    """

    @abstractmethod
    def process(self, result: TranscriptionResult) -> TranscriptionResult:
        """
        Transform a transcription result and return the processed version.

        Parameters
        ----------
        result : TranscriptionResult
            The transcription result produced by the ASR pipeline (or by the
            previous post-processor in the chain).

        Returns
        -------
        TranscriptionResult
            A new ``TranscriptionResult`` instance containing the processed
            text. Implementations should not mutate ``result`` in place;
            always return a fresh instance to preserve immutability.

        Raises
        ------
        ValueError
            If the result contains data that cannot be handled by this
            post-processor (e.g. unsupported language).
        RuntimeError
            If an external resource (model, API, etc.) fails during processing.
        """
        pass
