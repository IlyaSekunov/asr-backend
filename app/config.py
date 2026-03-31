"""
Central configuration for the speech processing pipeline.
All tuneable constants and environment-driven settings live here.
"""

from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # General
    ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav"})
    APP_TITLE: str = "Speech Processing API"
    APP_VERSION: str = "3.1.1"
    AUDIO_STORAGE_DIR: str = "/shared/audio"
    DEBUG: bool = False
    MAX_AUDIO_FILE_SIZE_MB: int = 500  # Hard upper limit for uploaded audio files

    # Model
    COMPUTE_DEVICE: ComputeDevice = ComputeDevice.CPU
    DENOISE_PROP_DECREASE: float = 0.9
    DENOISE_STATIONARY: bool = True
    DENOISING_ENABLED: bool = True
    LOUDNESS_METHOD: LoudnessNormalization = LoudnessNormalization.LUFS
    LOUDNESS_NORMALIZATION_ENABLED: bool = True
    LOUDNESS_TARGET: float = -23.0  # EBU R128 broadcast standard
    MODEL_SIZE: WhisperModelSize = WhisperModelSize.SMALL
    QUANTIZATION: Quantization = Quantization.INT8
    TARGET_SAMPLE_RATE: int = 16_000  # Whisper minimum required sample rate
    VAD_ENABLED: bool = True

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_QUEUE: str = "asr"
    REDIS_FAILED_JOBS_RETRY_COUNT: int = 3
    REDIS_QUEUE_RESULT_TTL: int = 300  # seconds to retain a successful result
    REDIS_QUEUE_FAILURE_TTL: int = 300  # seconds to retain a failed job for inspection
    REDIS_QUEUE_JOB_TIMEOUT: int = 1200  # 20 min

    # Metrics
    METRICS_PORT: int = 9091  # Prometheus scrape port on the worker
    METRICS_COLLECTION_INTERVAL: int = 15  # seconds between system-metrics samples

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()
