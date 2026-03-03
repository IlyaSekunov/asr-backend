"""Utility helpers for task management."""

import uuid


def generate_task_id() -> str:
    """Return a new random UUID string suitable for use as a task identifier."""
    return str(uuid.uuid4())
