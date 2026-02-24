"""Unit tests for AI services."""
import pytest
from pathlib import Path
from PIL import Image
from app.services.ai_service import (
    BackgroundRemovalService,
    SceneGenerationService,
    RelightingService,
    AIServiceFactory,
)


class TestBackgroundRemovalService:
    """Tests for BackgroundRemovalService."""

    @pytest.fixture
    def service(self) -> BackgroundRemovalService:
        """Create service instance without API URL (local mode)."""
        return BackgroundRemovalService(api_url=None)

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> str:
        """Create a sample image for testing."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)
        return str(img_path)

    @pytest.mark.asyncio
    async def test_local_processing(
        self, service: BackgroundRemovalService, sample_image: str
    ) -> None:
        """Test local background removal creates output."""
        result = await service.process(sample_image)

        assert Path(result).exists()
        assert result.endswith("bg_removed.png")

        # Check output is RGBA
        output_img = Image.open(result)
        assert output_img.mode == "RGBA"


class TestSceneGenerationService:
    """Tests for SceneGenerationService."""

    @pytest.fixture
    def service(self) -> SceneGenerationService:
        """Create service instance without API URL (local mode)."""
        return SceneGenerationService(api_url=None)

    @pytest.fixture
    def sample_image(self, tmp_path: Path) -> str:
        """Create a sample image for testing."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGBA", (100, 100), color="red")
        img.save(img_path)
        return str(img_path)

    @pytest.mark.asyncio
    async def test_local_processing(
        self, service: SceneGenerationService, sample_image: str
    ) -> None:
        """Test local scene generation creates output."""
        result = await service.process(sample_image, prompt="test scene")

        assert Path(result).exists()
        assert result.endswith("scene.png")


class TestRelightingService:
    """Tests for RelightingService."""

    @pytest.fixture
    def service(self) -> RelightingService:
        """Create service instance without API URL (placeholder mode)."""
        return RelightingService(api_url=None, use_local_model=False)

    @pytest.fixture
    def sample_images(self, tmp_path: Path) -> tuple[str, str]:
        """Create sample foreground and background images."""
        fg_path = tmp_path / "fg.png"
        bg_path = tmp_path / "bg.png"

        fg = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
        bg = Image.new("RGB", (100, 100), color="blue")

        fg.save(fg_path)
        bg.save(bg_path)

        return str(fg_path), str(bg_path)

    @pytest.mark.asyncio
    async def test_local_processing(
        self, service: RelightingService, sample_images: tuple[str, str]
    ) -> None:
        """Test relighting creates composited output."""
        fg_path, bg_path = sample_images
        result = await service.process(fg_path, background_path=bg_path)

        assert Path(result).exists()
        assert result.endswith("relit.png")

    @pytest.mark.asyncio
    async def test_local_processing_no_background(
        self, service: RelightingService, sample_images: tuple[str, str]
    ) -> None:
        """Test relighting without background."""
        fg_path, _ = sample_images
        result = await service.process(fg_path, background_path="")

        assert Path(result).exists()


class TestAIServiceFactory:
    """Tests for AIServiceFactory."""

    def test_get_background_removal_service(self) -> None:
        """Test factory returns BackgroundRemovalService."""
        service = AIServiceFactory.get_background_removal_service()
        assert isinstance(service, BackgroundRemovalService)

    def test_get_scene_generation_service(self) -> None:
        """Test factory returns SceneGenerationService."""
        service = AIServiceFactory.get_scene_generation_service()
        assert isinstance(service, SceneGenerationService)

    def test_get_relighting_service(self) -> None:
        """Test factory returns RelightingService."""
        service = AIServiceFactory.get_relighting_service()
        assert isinstance(service, RelightingService)
