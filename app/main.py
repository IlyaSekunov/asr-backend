"""
FastAPI application factory.

Import ``app`` from this module to run with Uvicorn:

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from rq_dashboard_fast import RedisQueueDashboard

from app.api.routes.transcription import router as transcription_router
from app.config import settings


def _configure_logging() -> None:
    """Configure Loguru: JSON in production, human-readable with colour in debug mode."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="DEBUG" if settings.DEBUG else "INFO",
        format=(
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | "
            "{name}:{function}:{line} | {message}"
        ),
        serialize=not settings.DEBUG,
        colorize=settings.DEBUG,
    )


def _mount_metrics(application: FastAPI) -> None:
    """
    Mount a /metrics endpoint and auto-instrument all HTTP routes.

    The endpoint is served as a separate ASGI sub-application so that
    Prometheus scrapes never appear in the FastAPI request metrics.
    """
    metrics_app = make_asgi_app()
    application.mount("/metrics", metrics_app)

    Instrumentator(
        should_group_status_codes=False,
        excluded_handlers=["/metrics", "/docs", "/openapi.json"],
    ).instrument(application).expose(application, include_in_schema=False)


def _mount_rq_dashboard(application: FastAPI) -> None:
    """
    Mount a /rq endpoint for RQ dashboard.
    """
    dashboard = RedisQueueDashboard(
        redis_url=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/",
        prefix="/rq",
    )
    application.mount("/rq", dashboard)


def create_app() -> FastAPI:
    """Assemble and return the FastAPI application."""
    _configure_logging()

    application = FastAPI(
        title=settings.APP_TITLE,
        version=settings.APP_VERSION,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(transcription_router, prefix="/api/v1")
    _mount_metrics(application)
    _mount_rq_dashboard(application)

    logger.info(
        "Application created | title='{}' version={}",
        settings.APP_TITLE,
        settings.APP_VERSION,
    )
    return application


app = create_app()
