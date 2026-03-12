"""High-level helpers for enqueuing jobs and querying their state via RQ."""

from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus, Retry

from app.asyncqueue.redis_queue import redis_connection, redis_queue
from app.config import settings
from app.schemas.transcription import TaskStatus
from app.transcribers.transcription_result import TranscriptionResult

# Resolved by RQ as an import path inside the worker process only.
# Do NOT replace this with a direct `from app.asyncqueue.tasks import transcribe_task`.
# Importing tasks.py triggers `_asr_pipeline = create_asr_pipeline()` at module level,
# which loads the Whisper model. That must happen in the worker process, not here —
# this module is imported by the API process via the route layer.
_TRANSCRIBE_TASK_PATH = "app.asyncqueue.tasks.transcribe_task"

# Maps RQ job statuses to the API's TaskStatus enum.
_JOB_STATUS_MAP: dict[JobStatus, TaskStatus] = {
    JobStatus.FINISHED: TaskStatus.READY,
    JobStatus.STARTED: TaskStatus.STARTED,
    JobStatus.FAILED: TaskStatus.FAILED,
}


def _map_job_status(status: JobStatus) -> TaskStatus:
    """Return the API TaskStatus for a given RQ JobStatus, defaulting to QUEUED."""
    return _JOB_STATUS_MAP.get(status, TaskStatus.QUEUED)


def job_exists(job_id: str) -> bool:
    try:
        Job.fetch(id=job_id, connection=redis_connection)
        return True
    except NoSuchJobError:
        return False


def fetch_job_status(job_id: str) -> TaskStatus:
    job = Job.fetch(id=job_id, connection=redis_connection)
    return _map_job_status(job.get_status())


def fetch_job_result(job_id: str) -> TranscriptionResult | None:
    job = Job.fetch(id=job_id, connection=redis_connection)
    return job.return_value()


def delete_job(job_id: str) -> None:
    job = Job.fetch(id=job_id, connection=redis_connection)
    job.delete()


def enqueue_transcription_task(file_path: str, task_id: str) -> None:
    redis_queue.enqueue(
        _TRANSCRIBE_TASK_PATH,
        file_path,
        job_id=task_id,
        result_ttl=settings.REDIS_QUEUE_RESULT_TTL,
        failure_ttl=settings.REDIS_QUEUE_FAILURE_TTL,
        job_timeout=settings.REDIS_QUEUE_JOB_TIMEOUT,
        retry=Retry(max=settings.REDIS_FAILED_JOBS_RETRY_COUNT),
    )
