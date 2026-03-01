from rq import Worker

from app.asyncqueue.redis_queue import redis_queue, redis_connection

if __name__ == "__main__":
    worker = Worker(
        queues=[redis_queue],
        connection=redis_connection,
    )
    worker.work()
