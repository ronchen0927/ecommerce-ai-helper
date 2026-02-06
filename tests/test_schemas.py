"""Unit tests for task schemas."""
from datetime import datetime, timezone
from app.schemas.task import (
    TaskStatus,
    TaskCreate,
    TaskResponse,
    TaskStatusResponse,
    UploadResponse,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_status_values(self) -> None:
        """Test all status values exist."""
        assert TaskStatus.PENDING == "PENDING"
        assert TaskStatus.REMOVING_BG == "REMOVING_BG"
        assert TaskStatus.GENERATING_SCENE == "GENERATING_SCENE"
        assert TaskStatus.RELIGHTING == "RELIGHTING"
        assert TaskStatus.COMPLETED == "COMPLETED"
        assert TaskStatus.FAILED == "FAILED"

    def test_status_count(self) -> None:
        """Test correct number of statuses."""
        assert len(TaskStatus) == 6


class TestTaskCreate:
    """Tests for TaskCreate schema."""

    def test_create_with_prompt(self) -> None:
        """Test creating task with scene prompt."""
        task = TaskCreate(scene_prompt="studio lighting")
        assert task.scene_prompt == "studio lighting"

    def test_create_without_prompt(self) -> None:
        """Test creating task without scene prompt."""
        task = TaskCreate(scene_prompt=None)
        assert task.scene_prompt is None


class TestTaskResponse:
    """Tests for TaskResponse schema."""

    def test_task_response_fields(self) -> None:
        """Test task response contains all required fields."""
        now = datetime.now(timezone.utc)
        response = TaskResponse(
            task_id="test-123",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        assert response.task_id == "test-123"
        assert response.status == TaskStatus.PENDING
        assert response.result_url is None
        assert response.error is None

    def test_task_response_with_result(self) -> None:
        """Test task response with result URL."""
        now = datetime.now(timezone.utc)
        response = TaskResponse(
            task_id="test-123",
            status=TaskStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            result_url="/storage/final.png",
        )
        assert response.result_url == "/storage/final.png"


class TestTaskStatusResponse:
    """Tests for TaskStatusResponse schema."""

    def test_status_response(self) -> None:
        """Test status response fields."""
        response = TaskStatusResponse(
            task_id="test-123",
            status=TaskStatus.REMOVING_BG,
            progress="Removing background...",
        )
        assert response.task_id == "test-123"
        assert response.status == TaskStatus.REMOVING_BG
        assert response.progress == "Removing background..."


class TestUploadResponse:
    """Tests for UploadResponse schema."""

    def test_upload_response(self) -> None:
        """Test upload response fields."""
        response = UploadResponse(
            task_id="test-123",
            message="Image uploaded",
            status=TaskStatus.PENDING,
        )
        assert response.task_id == "test-123"
        assert response.message == "Image uploaded"
        assert response.status == TaskStatus.PENDING
