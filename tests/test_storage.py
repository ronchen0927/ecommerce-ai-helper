"""Unit tests for storage service."""
import pytest
from pathlib import Path
from app.services.storage import LocalStorage


class TestLocalStorage:
    """Tests for LocalStorage service."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorage:
        """Create a LocalStorage instance with temp directory."""
        return LocalStorage(base_dir=tmp_path)

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> str:
        """Create a sample file for upload testing."""
        file_path = tmp_path / "sample.txt"
        file_path.write_text("test content")
        return str(file_path)

    @pytest.mark.asyncio
    async def test_upload(self, storage: LocalStorage, sample_file: str) -> None:
        """Test file upload."""
        result = await storage.upload(sample_file, "uploads/test.txt")

        assert Path(result).exists()
        assert Path(result).read_text() == "test content"

    @pytest.mark.asyncio
    async def test_upload_creates_directories(
        self, storage: LocalStorage, sample_file: str
    ) -> None:
        """Test upload creates parent directories."""
        result = await storage.upload(sample_file, "deep/nested/path/test.txt")

        assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_download(
        self, storage: LocalStorage, sample_file: str, tmp_path: Path
    ) -> None:
        """Test file download."""
        # First upload
        await storage.upload(sample_file, "uploads/test.txt")

        # Then download
        download_path = tmp_path / "downloaded.txt"
        await storage.download("uploads/test.txt", str(download_path))

        assert download_path.exists()
        assert download_path.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_download_not_found(self, storage: LocalStorage) -> None:
        """Test download raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            await storage.download("nonexistent.txt", "output.txt")

    def test_get_url(self, storage: LocalStorage) -> None:
        """Test get_url returns correct path."""
        url = storage.get_url("uploads/test.txt")
        # Use Path for cross-platform comparison
        assert Path(url).name == "test.txt"
        assert "uploads" in url or "uploads" in str(Path(url).parent)
