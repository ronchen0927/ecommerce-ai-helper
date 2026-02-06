"""
AI Service module for image processing pipeline.

This module provides async interfaces to various AI models:
- RMBG-1.4 for background removal (local model first, API fallback)
- Flux/SDXL for scene generation
- IC-Light for relighting
"""
# Standard library imports
import base64
import logging
from pathlib import Path
from typing import Any, Optional

# Third-party imports
import httpx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pydantic import BaseModel, ConfigDict
from skimage import io as skimage_io
from torchvision.transforms.functional import normalize
from transformers import AutoModelForImageSegmentation

# Local imports
from app.core.config import get_settings

# Module-level state for lazy model loading
_rmbg_model: Any = None
_torch_device: Any = None

logger = logging.getLogger(__name__)


class AIServiceError(Exception):
    """Base exception for AI service errors."""
    pass


class AIService(BaseModel):
    """Base class for AI services using Pydantic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Process an image and return the path to the result.

        Args:
            image_path: Path to the input image.
            **kwargs: Additional parameters for the specific service.

        Returns:
            Path to the processed image.
        """
        raise NotImplementedError


def _load_rmbg_model() -> tuple[Any, Any]:
    """
    Lazy load RMBG-1.4 model and get device.

    Returns:
        Tuple of (model, device)
    """
    global _rmbg_model, _torch_device

    if _rmbg_model is not None:
        return _rmbg_model, _torch_device

    _torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading RMBG-1.4 model on {_torch_device}...")
    _rmbg_model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-1.4",
        trust_remote_code=True
    )
    _rmbg_model.to(_torch_device)
    _rmbg_model.eval()
    logger.info("RMBG-1.4 model loaded successfully")

    return _rmbg_model, _torch_device


class BackgroundRemovalService(AIService):
    """
    Background removal service using RMBG-1.4.

    Strategy: Local model first → API fallback
    - Tries local GPU inference first (RMBG-1.4)
    - Falls back to API if local processing fails
    """

    api_url: Optional[str] = None
    use_local_model: bool = True

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.rmbg_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Remove background from an image.

        Strategy: Local first → API fallback

        Args:
            image_path: Path to the input image.

        Returns:
            Path to the image with background removed (PNG with transparency).
        """
        # Try local model first
        if self.use_local_model:
            try:
                return await self._process_local_model(image_path)
            except Exception as e:
                import traceback
                logger.warning(f"Local RMBG model failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        # Fallback to API
        if self.api_url:
            try:
                return await self._process_api(image_path)
            except Exception as e:
                logger.error(f"API fallback also failed: {e}")
                raise AIServiceError(f"Background removal failed: {e}") from e

        # Last resort: simple placeholder processing
        logger.warning("No API configured, using placeholder processing")
        return await self._process_placeholder(image_path)

    async def _process_local_model(self, image_path: str) -> str:
        """
        Process using local RMBG-1.4 model on GPU.

        Uses IS-Net architecture with 1024x1024 input size.
        """
        # Load model (lazy loading)
        model, device = _load_rmbg_model()

        # Read image
        orig_im = skimage_io.imread(image_path)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        # Preprocess
        image_tensor = self._preprocess_image(orig_im, model_input_size)
        image_tensor = image_tensor.to(device)

        # Inference
        with torch.no_grad():
            result = model(image_tensor)

        # Postprocess
        mask = self._postprocess_mask(result[0][0], orig_im_size)

        # Apply mask to original image
        output_path = Path(image_path).parent / "bg_removed.png"
        pil_mask = Image.fromarray(mask)
        orig_image = Image.open(image_path).convert("RGBA")

        # Create transparent image
        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask)
        no_bg_image.save(output_path, "PNG")

        logger.info(f"Background removed successfully using local model: {output_path}")
        return str(output_path)

    def _preprocess_image(
        self,
        im: np.ndarray,
        model_input_size: list[int],
    ) -> torch.Tensor:
        """Preprocess image for RMBG-1.4 model."""
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]

        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        im_tensor = F.interpolate(
            torch.unsqueeze(im_tensor, 0),
            size=model_input_size,
            mode='bilinear'
        )
        image = torch.divide(im_tensor, 255.0)
        image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        return torch.as_tensor(image)

    def _postprocess_mask(
        self,
        result: torch.Tensor,
        im_size: tuple[int, int],
    ) -> np.ndarray:
        """Postprocess model output to get mask."""
        # result shape is [C, H, W] = [1, 1024, 1024]
        # F.interpolate needs [N, C, H, W]
        if result.dim() == 3:
            result = result.unsqueeze(0)  # [1, 1, 1024, 1024]

        # Resize to original image size
        result = F.interpolate(result, size=im_size, mode='bilinear', align_corners=False)

        # Normalize to 0-255
        result = result.squeeze()  # Remove batch and channel dims -> [H, W]
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)

        # Convert to numpy uint8
        im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
        return np.asarray(im_array)

    async def _process_api(self, image_path: str) -> str:
        """Process using remote API."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Encode image as base64 for API
            image_b64 = base64.b64encode(image_data).decode("utf-8")

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json={"image": image_b64},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise AIServiceError(
                    f"RMBG API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            result_image_b64 = result.get("result", result.get("image", ""))

            # Save result
            output_path = Path(image_path).parent / "bg_removed.png"
            image_bytes = base64.b64decode(result_image_b64)
            with open(output_path, "wb") as f:
                f.write(image_bytes)

            return str(output_path)

    async def _process_placeholder(self, image_path: str) -> str:
        """
        Placeholder processing for testing when no model/API available.
        Simply converts image to RGBA.
        """
        output_path = Path(image_path).parent / "bg_removed.png"

        img = Image.open(image_path)
        output_img = img.convert("RGBA") if img.mode != "RGBA" else img
        output_img.save(output_path, "PNG")

        return str(output_path)


class SceneGenerationService(AIService):
    """
    Scene generation service using Flux or SDXL.

    Generates new backgrounds/scenes based on text prompts.
    """

    api_url: Optional[str] = None

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.flux_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Generate a scene based on the input image and prompt.

        Args:
            image_path: Path to the foreground image (with transparent background).
            prompt: Text prompt describing the desired scene.

        Returns:
            Path to the generated scene image.
        """
        prompt = kwargs.get("prompt", "professional product photography, studio lighting")

        if not self.api_url:
            return await self._process_local(image_path, prompt)

        return await self._process_api(image_path, prompt)

    async def _process_api(self, image_path: str, prompt: str) -> str:
        """Process using remote API (Flux/SDXL)."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            with open(image_path, "rb") as f:
                image_data = f.read()

            image_b64 = base64.b64encode(image_data).decode("utf-8")

            payload = {
                "image": image_b64,
                "prompt": prompt,
                "negative_prompt": "blurry, low quality, distorted",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise AIServiceError(
                    f"Scene generation API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            result_image_b64 = result.get("result", result.get("image", ""))

            output_path = Path(image_path).parent / "scene.png"
            image_bytes = base64.b64decode(result_image_b64)
            with open(output_path, "wb") as f:
                f.write(image_bytes)

            return str(output_path)

    async def _process_local(self, image_path: str, prompt: str) -> str:
        """
        Local processing placeholder.

        In production, this would use Flux/SDXL model directly.
        """
        output_path = Path(image_path).parent / "scene.png"

        # Create a simple gradient background for testing
        img = Image.new("RGB", (1024, 1024), color=(240, 240, 245))
        img.save(output_path, "PNG")

        return str(output_path)


class RelightingService(AIService):
    """
    Relighting service using IC-Light.

    IC-Light harmonizes lighting between foreground and background.
    """

    api_url: Optional[str] = None

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.iclight_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Apply relighting to composite foreground and background.

        Args:
            image_path: Path to the foreground image.
            background_path: Path to the background/scene image.

        Returns:
            Path to the final composited image with harmonized lighting.
        """
        background_path = kwargs.get("background_path", "")

        if not self.api_url:
            return await self._process_local(image_path, background_path)

        return await self._process_api(image_path, background_path)

    async def _process_api(self, fg_path: str, bg_path: str) -> str:
        """Process using remote IC-Light API."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            with open(fg_path, "rb") as f:
                fg_data = base64.b64encode(f.read()).decode("utf-8")
            with open(bg_path, "rb") as f:
                bg_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "foreground": fg_data,
                "background": bg_data,
                "lighting_mode": "auto",
            }

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise AIServiceError(
                    f"IC-Light API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            result_image_b64 = result.get("result", result.get("image", ""))

            output_path = Path(fg_path).parent / "final.png"
            image_bytes = base64.b64decode(result_image_b64)
            with open(output_path, "wb") as f:
                f.write(image_bytes)

            return str(output_path)

    async def _process_local(self, fg_path: str, bg_path: str) -> str:
        """
        Local processing placeholder.

        In production, this would use IC-Light model directly.
        For now, this does a simple composite.
        """
        output_path = Path(fg_path).parent / "final.png"

        # Simple composite for testing
        fg_img = Image.open(fg_path)
        if bg_path and Path(bg_path).exists():
            bg_img = Image.open(bg_path)
            resized_bg = bg_img.resize(fg_img.size)

            if fg_img.mode == "RGBA":
                bg_rgba = resized_bg.convert("RGBA")
                result = Image.alpha_composite(bg_rgba, fg_img)
            else:
                result = fg_img
        else:
            result = fg_img

        result.save(output_path, "PNG")
        return str(output_path)


class AIServiceFactory:
    """Factory for creating AI service instances."""

    @staticmethod
    def get_background_removal_service(use_local: bool = True) -> BackgroundRemovalService:
        """Get background removal service instance."""
        return BackgroundRemovalService(use_local_model=use_local)

    @staticmethod
    def get_scene_generation_service() -> SceneGenerationService:
        """Get scene generation service instance."""
        return SceneGenerationService()

    @staticmethod
    def get_relighting_service() -> RelightingService:
        """Get relighting service instance."""
        return RelightingService()
