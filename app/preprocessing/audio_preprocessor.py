from abc import ABC, abstractmethod

import numpy as np


class AudioPreprocessor(ABC):
    """
    Abstract base class for audio preprocessing operations.

    This interface defines the contract for all audio preprocessors in the pipeline.
    Each concrete implementation must provide a process() method that transforms
    an input audio signal according to specific preprocessing logic (e.g., noise
    reduction, resampling, normalization).

    The class follows the Template Method pattern, ensuring consistent integration
    of preprocessing steps while allowing flexible implementation details.

    All subclasses must:
        - Inherit from this base class
        - Implement the process() method
        - Maintain the expected audio format (normalized values in [-1.0, 1.0])
        - Preserve the sample rate unless explicitly modified

    Attributes
    ----------
    None (abstract base class - no common attributes defined)

    Methods
    -------
    process(audio)
        Transform input audio and return processed version.
    """

    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformation to an audio signal.

        This method must be implemented by all concrete subclasses to define
        the specific preprocessing operation. Implementations should:
            - Accept audio normalized to [-1.0, 1.0] range
            - Return audio in the same format unless explicitly documented
            - Handle edge cases gracefully (empty input, invalid values)
            - Preserve audio length unless the transformation specifically
              modifies it (e.g., resampling, trimming)

        Parameters
        ----------
        audio : np.ndarray
            Input audio time series with values normalized to [-1.0, 1.0].
            Shape can be (n_samples,) for mono or (n_channels, n_samples)
            for multi-channel audio.

        Returns
        -------
        np.ndarray
            Processed audio signal. Should maintain the same dtype and
            value range as input unless explicitly documented otherwise.

        Raises
        ------
        ValueError
            If the input audio format is invalid or preprocessing fails.
        NotImplementedError
            If called directly on the base class.

        Notes
        -----
        Subclasses should document any deviations from the expected
        input/output contract (e.g., changing sample rate, modifying
        length, returning different value ranges).
        """
        pass