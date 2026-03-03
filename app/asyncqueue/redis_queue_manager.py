from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus

from app.asyncqueue.redis_queue import redis_connection, redis_queue
from app.config import settings
from app.schemas.transcription import TaskStatus
from app.transcribers.transcription_result import TranscriptionResult


def _map_job_status(status: JobStatus) -> TaskStatus:
    if status in [JobStatus.FAILED]:
        return TaskStatus.FAILED

    if status in [JobStatus.FINISHED]:
        return TaskStatus.READY

    if status in [JobStatus.STARTED]:
        return TaskStatus.STARTED

    return TaskStatus.QUEUED


def job_exists(job_id: str) -> bool:
    try:
        Job.fetch(id=job_id, connection=redis_connection)
        return True
    except NoSuchJobError:
        return False


def fetch_job_status(job_id: str) -> TaskStatus:
    job = Job.fetch(id=job_id, connection=redis_connection)
    status = job.get_status()
    return _map_job_status(status)


def fetch_job_result(job_id: str) -> TranscriptionResult | None:
    job = Job.fetch(id=job_id, connection=redis_connection)
    return job.return_value()


def delete_job(job_id: str) -> None:
    job = Job.fetch(id=job_id, connection=redis_connection)
    job.delete()


def enqueue_transcription_task(file_to_audio: str, task_id: str) -> None:
    redis_queue.enqueue(
        settings.TRANSCRIBE_FUNCTION_TASK_PATH,
        file_to_audio,
        job_id=task_id,
        result_ttl=settings.REDIS_QUEUE_RESULT_TTL,
        failure_ttl=settings.REDIS_QUEUE_FAILURE_TTL,
    )
