"""Worker entry-point.

Uses RQ's ``SimpleWorker`` which executes jobs in the same process without
fork().  This allows the Whisper model to be loaded once at startup and
reused across all jobs — fork() + multithreaded model code (ONNX Runtime)
causes deadlocks in child processes.
"""

import app.asyncqueue.tasks  # noqa: F401  — loads _asr_pipeline eagerly

from prometheus_client import start_http_server
from rq.worker import SimpleWorker

from app.asyncqueue.redis_queue import redis_connection, redis_queue
from app.config import settings
from app.metrics.system_collector import start_system_metrics_collector

if __name__ == "__main__":
    start_http_server(settings.METRICS_PORT)
    start_system_metrics_collector(interval=settings.METRICS_COLLECTION_INTERVAL)

    worker = SimpleWorker(
        queues=[redis_queue],
        connection=redis_connection,
    )
    worker.work()
