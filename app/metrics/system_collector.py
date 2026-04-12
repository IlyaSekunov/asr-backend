"""Background system-metrics collector for the ASR worker process.

Runs a daemon thread that samples CPU, RAM and GPU utilisation at a
configurable interval and publishes them as Prometheus Gauges.

CPU utilisation is normalised to 0–100% of total machine capacity by
dividing psutil's per-core sum by the logical CPU count.

GPU collection relies on ``pynvml`` (nvidia-ml-py3).  If the library is
not installed, or if NVML cannot be initialised (e.g. CPU-only host),
GPU metrics are silently skipped and only CPU/RAM are reported.
"""

from __future__ import annotations

import os
import threading
import time

import psutil
from loguru import logger

from app.metrics.metrics import (
    worker_cpu_percent,
    worker_gpu_memory_total_bytes,
    worker_gpu_memory_used_bytes,
    worker_gpu_utilization_percent,
    worker_memory_bytes,
)

try:
    import pynvml as _pynvml

    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


def _init_gpu() -> int:
    """Initialise NVML and return the number of GPUs found (0 on failure)."""
    if not _PYNVML_AVAILABLE:
        logger.info("pynvml not installed — GPU metrics disabled.")
        return 0
    try:
        _pynvml.nvmlInit()
        count = _pynvml.nvmlDeviceGetCount()
        logger.info("GPU metrics enabled: {} device(s) detected.", count)
        return count
    except Exception as exc:
        logger.info("GPU metrics disabled (NVML init failed: {}).", exc)
        return 0


def _collect_loop(
        process: psutil.Process,
        gpu_count: int,
        interval: int,
        cpu_count: int,
) -> None:
    """Main collection loop — never raises; logs warnings on individual errors."""
    # Warm-up call so the first cpu_percent() reading is meaningful.
    process.cpu_percent(interval=None)

    while True:
        time.sleep(interval)

        # CPU & Memory
        try:
            # psutil returns per-core sum (e.g. 260% on 4 cores).
            # Dividing by cpu_count normalises to 0–100% of total CPU.
            raw_cpu = process.cpu_percent(interval=None)
            worker_cpu_percent.set(raw_cpu / cpu_count)
            worker_memory_bytes.set(process.memory_info().rss)
        except Exception as exc:
            logger.warning("CPU/memory metric collection error: {}", exc)

        # GPU
        for idx in range(gpu_count):
            label = f"gpu{idx}"
            try:
                handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)

                mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
                worker_gpu_memory_used_bytes.labels(device=label).set(mem.used)
                worker_gpu_memory_total_bytes.labels(device=label).set(mem.total)

                util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                # pynvml already returns 0–100% for GPU utilisation.
                worker_gpu_utilization_percent.labels(device=label).set(util.gpu)
            except Exception as exc:
                logger.warning("GPU {} metric collection error: {}", label, exc)


def start_system_metrics_collector(interval: int = 15) -> None:
    """
    Start the background metrics-collection thread.

    Parameters
    ----------
    interval : int
        Sampling period in seconds (default: 15).
    """
    process = psutil.Process(os.getpid())
    gpu_count = _init_gpu()

    # Logical CPU count for normalising psutil's per-core CPU percent.
    cpu_count = psutil.cpu_count(logical=True) or 1

    thread = threading.Thread(
        target=_collect_loop,
        args=(process, gpu_count, interval, cpu_count),
        daemon=True,
        name="system-metrics-collector",
    )
    thread.start()
    logger.info(
        "System metrics collector started | interval={}s gpu_devices={} cpu_count={}",
        interval,
        gpu_count,
        cpu_count,
    )
