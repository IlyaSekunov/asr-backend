"""Redis connection and RQ queue used by both the API and the worker."""

from redis import Redis
from rq import Queue

from app.config import settings

redis_connection = Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
)

redis_queue = Queue(
    name=settings.REDIS_QUEUE,
    connection=redis_connection,
)
