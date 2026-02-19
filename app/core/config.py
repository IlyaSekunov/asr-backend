"""
Central configuration for the speech processing pipeline.
All tuneable constants and environment-driven settings live here.
"""

from enum import Enum

from pydantic_settings import BaseSettings


class ComputeDevice(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class WhisperModelSize(str, Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


class Quantization(str, Enum):
    INT8 = "int8"
    INT8_FLOAT16 = "int8_float16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class LoudnessNormalization(str, Enum):
    LUFS = "lufs"
    PEAK = "peak"
    RMS = "rms"


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_TITLE: str = "Speech Processing API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_SIZE: WhisperModelSize = WhisperModelSize.LARGE_V3
    QUANTIZATION: Quantization = Quantization.INT8
    COMPUTE_DEVICE: ComputeDevice = ComputeDevice.CPU

    VAD_ENABLED: bool = True

    DENOISING_ENABLED: bool = True
    DENOISE_PROP_DECREASE: float = 0.9
    DENOISE_STATIONARY: bool = True

    LOUDNESS_NORMALIZATION_ENABLED: bool = True
    LOUDNESS_METHOD: LoudnessNormalization = LoudnessNormalization.LUFS
    LOUDNESS_TARGET: float = -23.0

    TARGET_SAMPLE_RATE: int = 22_050

    ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav"})

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
