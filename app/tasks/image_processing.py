"""
Celery tasks for image processing pipeline.

This module contains the main image processing task that orchestrates
the three-stage AI pipeline:
1. Background Removal (RMBG-1.4)
2. Scene Generation (Flux/SDXL)
3. Relighting (IC-Light)
"""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Coroutine, Optional, TypeVar

from celery import Task

from app.core.celery_app import celery_app
from app.schemas.task import TaskStatus
from app.services.ai_service import AIServiceFactory
from app.services.storage import GCSStorage, LocalStorage, StorageService
from app.core.config import get_settings

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
    Main image processing pipeline (Hybrid GCS/Local Support).
    """
    settings = get_settings()
    # Initialize storage based on the path or settings
    if image_path.startswith("gs://") or settings.storage_type == "gcs":
        storage = GCSStorage(bucket_name=settings.gcs_bucket_name)
    else:
        storage = LocalStorage(base_dir=Path(settings.local_storage_path))

    # Create a local temporary directory for this task
    temp_dir = Path(tempfile.mkdtemp(prefix=f"task_{task_id}_"))
    local_input_path = temp_dir / Path(image_path).name
    
    try:
        # 1. Download source image if it's remote or just copy it to temp
        if image_path.startswith("gs://") or image_path.startswith("http"):
            # If it's gs://, image_path is just the key for GCSStorage? 
            # Or is it the full URL? Routes.py passes 'stored_path'.
            # For GCS, upload returns 'gs://...'. Let's handle it.
            gcs_key = image_path.replace(f"gs://{settings.gcs_bucket_name}/", "")
            run_async(storage.download(gcs_key, str(local_input_path)))
        else:
            # If local, copy to temp
            shutil.copy2(image_path, local_input_path)

        # Stage 1: Background Removal
        self.update_task_status(task_id, TaskStatus.REMOVING_BG)
        bg_service = AIServiceFactory.get_background_removal_service()
        local_bg_removed = run_async(bg_service.process(str(local_input_path)))

        # Stage 2: Scene Generation
        self.update_task_status(task_id, TaskStatus.GENERATING_SCENE)
        scene_service = AIServiceFactory.get_scene_generation_service()
        prompt = scene_prompt or "professional product photography, studio lighting"
        local_scene = run_async(scene_service.process(local_bg_removed, prompt=prompt))

        # Stage 3: Relighting
        self.update_task_status(task_id, TaskStatus.RELIGHTING)
        relight_service = AIServiceFactory.get_relighting_service()
        local_final = run_async(
            relight_service.process(local_bg_removed, background_path=local_scene, prompt=prompt)
        )

        # 4. Upload final result back to storage
        final_dest_key = f"processed/{task_id}/relit.png"
        final_result_url = run_async(storage.upload(local_final, final_dest_key))

        # Mark as completed
        self.update_task_status(task_id, TaskStatus.COMPLETED)

        return {
            "task_id": task_id,
            "status": TaskStatus.COMPLETED.value,
            "result_url": final_result_url,
        }

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        self.update_task_status(task_id, TaskStatus.FAILED)
        return {
            "task_id": task_id,
            "status": TaskStatus.FAILED.value,
            "error": error_msg,
        }
    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
