"""Prometheus metric definitions for the ASR worker.

All metrics are defined here so they can be imported by both the task
module and the system collector without circular dependencies.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Transcription outcomes ─────────────────────────────────────────────────

transcription_success_total = Counter(
    "asr_transcription_success_total",
    "Total number of audio files successfully transcribed.",
)

transcription_failure_total = Counter(
    "asr_transcription_failure_total",
    "Total number of audio files that failed during transcription.",
)

# ── Audio characteristics ──────────────────────────────────────────────────

audio_duration_minutes = Histogram(
    "asr_audio_duration_minutes",
    "Duration of processed audio files in minutes.",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)

language_detections_total = Counter(
    "asr_language_detections_total",
    "Number of transcriptions per detected language.",
    labelnames=["language"],
)

# ── Performance ────────────────────────────────────────────────────────────

real_time_factor = Histogram(
    "asr_real_time_factor",
    "Real Time Factor: total_processing_seconds / audio_duration_seconds. "
    "RTF < 1.0 means faster-than-real-time; RTF > 1.0 is slower.",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0],
)

# ── Worker system resources ────────────────────────────────────────────────

worker_cpu_percent = Gauge(
    "asr_worker_cpu_percent",
    "Worker process CPU utilisation (percent of a single core).",
)

worker_memory_bytes = Gauge(
    "asr_worker_memory_bytes",
    "Worker process RSS memory usage in bytes.",
)

worker_gpu_utilization_percent = Gauge(
    "asr_worker_gpu_utilization_percent",
    "GPU compute utilisation in percent.",
    labelnames=["device"],
)

worker_gpu_memory_used_bytes = Gauge(
    "asr_worker_gpu_memory_used_bytes",
    "GPU memory currently in use, in bytes.",
    labelnames=["device"],
)

worker_gpu_memory_total_bytes = Gauge(
    "asr_worker_gpu_memory_total_bytes",
    "Total GPU memory available, in bytes.",
    labelnames=["device"],
)
