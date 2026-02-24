"""
Celery tasks for image processing pipeline.

This module contains the main image processing task that orchestrates
the three-stage AI pipeline:
1. Background Removal (RMBG-1.4)
2. Scene Generation (Flux/SDXL)
3. Relighting (IC-Light)
"""

import asyncio
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

from celery import Task

from app.core.celery_app import celery_app
from app.schemas.task import TaskStatus
from app.services.ai_service import AIServiceFactory

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Helper to run async code in sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class ImageProcessingTask(Task):  # type: ignore[misc]
    """Base task class for image processing with status updates."""

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status in the task store."""
        # In production, this would update Redis/DB
        # For now, we'll use Celery's result backend
        self.update_state(state=status.value, meta={"task_id": task_id})


@celery_app.task(
    bind=True,
    base=ImageProcessingTask,
    name="app.tasks.image_processing.process_image",
)
def process_image(
    self: ImageProcessingTask,
    task_id: str,
    image_path: str,
    scene_prompt: Optional[str] = None,
) -> dict[str, str]:
    """
    Main image processing pipeline.

    State Machine:
    PENDING -> REMOVING_BG -> GENERATING_SCENE -> RELIGHTING -> COMPLETED

    Args:
        task_id: Unique task identifier
        image_path: Path to the uploaded image
        scene_prompt: Optional prompt for scene generation
    """
    try:
        # Ensure output directory exists
        output_dir = Path("storage/processed") / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Background Removal
        self.update_task_status(task_id, TaskStatus.REMOVING_BG)
        bg_service = AIServiceFactory.get_background_removal_service()
        bg_removed_path = run_async(bg_service.process(image_path))

        # Stage 2: Scene Generation
        self.update_task_status(task_id, TaskStatus.GENERATING_SCENE)
        scene_service = AIServiceFactory.get_scene_generation_service()
        prompt = scene_prompt or "professional product photography, studio lighting"
        scene_path = run_async(scene_service.process(bg_removed_path, prompt=prompt))

        # Stage 3: Relighting
        self.update_task_status(task_id, TaskStatus.RELIGHTING)
        relight_service = AIServiceFactory.get_relighting_service()
        final_path = run_async(
            relight_service.process(bg_removed_path, background_path=scene_path)
        )

        # Mark as completed
        self.update_task_status(task_id, TaskStatus.COMPLETED)

        return {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED.value,
            "result_url": final_path,
        }

    except Exception as e:
        self.update_task_status(task_id, TaskStatus.FAILED)
        return {
            "task_id": task_id,
            "status": TaskStatus.FAILED.value,
            "error": str(e),
        }
