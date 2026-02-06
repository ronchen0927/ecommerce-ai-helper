"""Storage service interfaces and implementations."""
import shutil
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator


class StorageService(BaseModel):
    """Base class for storage services using Pydantic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def upload(self, file_path: str, destination_path: str) -> str:
        """
        Upload a file to storage.

        Args:
            file_path: Local path to the file to upload.
            destination_path: Path in the storage system.

        Returns:
            The public URL or internal path of the uploaded file.
        """
        raise NotImplementedError

    async def download(self, source_path: str, destination_path: str) -> None:
        """
        Download a file from storage.

        Args:
            source_path: Path in the storage system.
            destination_path: Local path to save the file.
        """
        raise NotImplementedError

    def get_url(self, path: str) -> str:
        """
        Get the accessible URL for a file in storage.

        Args:
            path: Path in the storage system.

        Returns:
            URL string.
        """
        raise NotImplementedError


class LocalStorage(StorageService):
    """Local file system implementation of StorageService."""

    base_dir: Path = Path("storage")

    @field_validator("base_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        return Path(v) if isinstance(v, str) else v

    def model_post_init(self, __context: object) -> None:
        """Create base directory after model initialization."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, file_path: str, destination_path: str) -> str:
        dest_full_path = self.base_dir / destination_path
        dest_full_path.parent.mkdir(parents=True, exist_ok=True)

        # In a real async context, we might use aiofiles, but shutil is fine for now for local
        # If strict async is needed, we'd offload this to a thread
        shutil.copy2(file_path, dest_full_path)
        return str(dest_full_path)

    async def download(self, source_path: str, destination_path: str) -> None:
        src_full_path = self.base_dir / source_path
        if not src_full_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")

        shutil.copy2(src_full_path, destination_path)

    def get_url(self, path: str) -> str:
        # For local storage, we might return a file path or a relative URL if served via static files
        return str(self.base_dir / path)


# Placeholder for GCS implementation
class GCSStorage(StorageService):
    """Google Cloud Storage implementation of StorageService."""

    bucket_name: str

    async def upload(self, file_path: str, destination_path: str) -> str:
        raise NotImplementedError("GCS upload not yet implemented")

    async def download(self, source_path: str, destination_path: str) -> None:
        raise NotImplementedError("GCS download not yet implemented")

    def get_url(self, path: str) -> str:
        return f"https://storage.googleapis.com/{self.bucket_name}/{path}"
