from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task processing status enum."""

    PENDING = "PENDING"
    REMOVING_BG = "REMOVING_BG"
    GENERATING_SCENE = "GENERATING_SCENE"
    RELIGHTING = "RELIGHTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskCreate(BaseModel):
    """Schema for creating a new task."""

    scene_prompt: Optional[str] = Field(None, description="Prompt for scene generation")
    negative_prompt: Optional[str] = Field(None, description="Negative prompt to exclude elements")


class TaskResponse(BaseModel):
    """Schema for task response."""

    task_id: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    original_url: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None

    class Config:
        from_attributes = True


class TaskStatusResponse(BaseModel):
    """Schema for task status query response."""

    task_id: str
    status: TaskStatus
    progress: Optional[str] = None
    result_url: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Schema for upload response."""

    task_id: str
    message: str
    status: TaskStatus
