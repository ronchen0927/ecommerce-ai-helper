"""Storage service interfaces and implementations."""

import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

from google.cloud import storage
from google.oauth2 import service_account
from pydantic import BaseModel, ConfigDict, field_validator

from app.core.config import get_settings


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

    async def delete(self, path: str) -> None:
        """
        Delete a file from storage.

        Args:
            path: Path in the storage system.
        """
        raise NotImplementedError

    async def exists(self, path: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            path: Path in the storage system.

        Returns:
            True if file exists.
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
        shutil.copy2(file_path, dest_full_path)
        return str(dest_full_path)

    async def download(self, source_path: str, destination_path: str) -> None:
        src_full_path = self.base_dir / source_path
        if not src_full_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        shutil.copy2(src_full_path, destination_path)

    def get_url(self, path: str) -> str:
        return str(self.base_dir / path)

    async def delete(self, path: str) -> None:
        file_path = self.base_dir / path
        if file_path.exists():
            file_path.unlink()

    async def exists(self, path: str) -> bool:
        return (self.base_dir / path).exists()


class GCSStorage(StorageService):
    """Google Cloud Storage implementation of StorageService."""

    bucket_name: str
    project_id: str = ""
    credentials_path: str = ""
    _client: Optional[Any] = None
    _bucket: Optional[Any] = None

    def model_post_init(self, __context: object) -> None:
        """Initialize GCS client after model initialization."""
        settings = get_settings()

        # Use settings if not provided
        if not self.project_id:
            self.project_id = settings.gcs_project_id
        if not self.credentials_path:
            self.credentials_path = settings.gcs_credentials_path

        # Initialize client
        if self.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self._client = storage.Client(
                project=self.project_id, credentials=credentials
            )
        else:
            # Use default credentials (e.g., from GOOGLE_APPLICATION_CREDENTIALS)
            self._client = storage.Client(project=self.project_id)

        self._bucket = self._client.bucket(self.bucket_name)

    async def upload(self, file_path: str, destination_path: str) -> str:
        """Upload file to GCS bucket."""
        blob = self._bucket.blob(destination_path)
        blob.upload_from_filename(file_path)
        return f"gs://{self.bucket_name}/{destination_path}"

    async def download(self, source_path: str, destination_path: str) -> None:
        """Download file from GCS bucket."""
        blob = self._bucket.blob(source_path)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {source_path}")
        blob.download_to_filename(destination_path)

    def get_url(self, path: str) -> str:
        """Get public URL for a file."""
        return f"https://storage.googleapis.com/{self.bucket_name}/{path}"

    def get_signed_url(
        self, path: str, expiration_minutes: int = 60, method: str = "GET"
    ) -> str:
        """
        Generate a signed URL for temporary access.

        Args:
            path: Path to the file in the bucket.
            expiration_minutes: URL expiration time in minutes.
            method: HTTP method (GET or PUT).

        Returns:
            Signed URL string.
        """
        blob = self._bucket.blob(path)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method=method,
        )
        return url

    async def delete(self, path: str) -> None:
        """Delete file from GCS bucket."""
        blob = self._bucket.blob(path)
        if blob.exists():
            blob.delete()

    async def exists(self, path: str) -> bool:
        """Check if file exists in GCS bucket."""
        blob = self._bucket.blob(path)
        return blob.exists()

    def list_files(self, prefix: str = "") -> list[str]:
        """
        List files in the bucket with optional prefix.

        Args:
            prefix: Filter files by prefix (folder path).

        Returns:
            List of file paths.
        """
        blobs = self._client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs]
