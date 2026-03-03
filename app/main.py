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

    logger.info(
        "Application created | title='{}' version={}",
        settings.APP_TITLE,
        settings.APP_VERSION,
    )
    return application


app = create_app()
