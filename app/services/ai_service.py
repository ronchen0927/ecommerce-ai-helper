"""
AI Service module for image processing pipeline.

This module provides async interfaces to various AI models:
- RMBG-1.4 for background removal (local model first, API fallback)
- Flux/SDXL for scene generation
- IC-Light for relighting (local model first, API fallback)
"""

# Standard library imports
import base64
import logging
import math
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

# IC-Light module-level state
_iclight_t2i_pipe: Any = None
_iclight_i2i_pipe: Any = None
_iclight_vae: Any = None
_iclight_device: Any = None

# Scene Generation module-level state
_scene_pipe: Any = None
_scene_device: Any = None

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
        "briaai/RMBG-1.4", trust_remote_code=True
    )
    _rmbg_model.to(_torch_device)
    _rmbg_model.eval()
    logger.info("RMBG-1.4 model loaded successfully")

    return _rmbg_model, _torch_device


def _load_scene_model() -> tuple[Any, Any]:
    """
    Lazy load SD 1.5 for local scene generation.
    """
    global _scene_pipe, _scene_device

    if _scene_pipe is not None:
        return _scene_pipe, _scene_device

    from diffusers import StableDiffusionPipeline
    import torch

    _scene_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading Local Scene Gen Model (SD 1.5) on {_scene_device}...")

    _scene_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    # Memory optimization
    _scene_pipe.enable_attention_slicing()

    logger.info("Local Scene Gen Model loaded successfully")
    return _scene_pipe, _scene_device


def _load_iclight_model() -> tuple[Any, Any, Any, Any]:
    """
    Lazy load IC-Light FBC model pipeline.

    Returns:
        Tuple of (t2i_pipe, i2i_pipe, vae, device)
    """
    global _iclight_t2i_pipe, _iclight_i2i_pipe, _iclight_vae, _iclight_device

    if _iclight_t2i_pipe is not None:
        return _iclight_t2i_pipe, _iclight_i2i_pipe, _iclight_vae, _iclight_device

    import safetensors.torch as sf
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from diffusers.models.attention_processor import AttnProcessor2_0
    from transformers import CLIPTextModel, CLIPTokenizer

    _iclight_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd15_name = "stablediffusionapi/realistic-vision-v51"

    logger.info(f"Loading IC-Light FBC model on {_iclight_device}...")

    # Load SD1.5 components
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")

    # Modify UNet conv_in: 4 -> 12 channels (fg latent + bg latent + noise)
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(
            12,
            unet.conv_in.out_channels,
            unet.conv_in.kernel_size,
            unet.conv_in.stride,
            unet.conv_in.padding,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in

    # Hook UNet forward to concatenate condition latents
    unet_original_forward = unet.forward

    def hooked_unet_forward(
        sample: Any, timestep: Any, encoder_hidden_states: Any, **kwargs: Any
    ) -> Any:
        # Avoid modifying kwargs in-place for multi-call stability
        ca_kwargs = kwargs.get("cross_attention_kwargs", {}).copy()
        c_concat = ca_kwargs.pop("concat_conds", None)

        if c_concat is not None:
            c_concat = c_concat.to(sample)
            c_concat = torch.cat(
                [c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0
            )
            sample = torch.cat([sample, c_concat], dim=1)

        return unet_original_forward(
            sample, timestep, encoder_hidden_states, cross_attention_kwargs=ca_kwargs
        )

    unet.forward = hooked_unet_forward

    # Download and merge IC-Light weights
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "iclight_sd15_fbc.safetensors"

    if not model_path.exists():
        logger.info("Downloading IC-Light FBC weights...")
        from torch.hub import download_url_to_file

        download_url_to_file(
            url="https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors",
            dst=str(model_path),
        )

    # Merge IC-Light offset weights into UNet
    sd_offset = sf.load_file(str(model_path))
    sd_origin = unet.state_dict()
    sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
    unet.load_state_dict(sd_merged, strict=True)
    del sd_offset, sd_origin, sd_merged

    # Move to device with float16
    text_encoder = text_encoder.to(device=_iclight_device, dtype=torch.float16)
    vae = vae.to(device=_iclight_device, dtype=torch.float16)
    unet = unet.to(device=_iclight_device, dtype=torch.float16)

    # Use SDP attention
    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # Create scheduler
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="sde-dpmsolver++",
        use_karras_sigmas=True,
        steps_offset=1,
    )

    # Create pipelines
    pipe_kwargs: dict[str, Any] = dict(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None,
    )

    _iclight_t2i_pipe = StableDiffusionPipeline(**pipe_kwargs).to(_iclight_device)
    _iclight_i2i_pipe = StableDiffusionImg2ImgPipeline(**pipe_kwargs).to(_iclight_device)
    _iclight_vae = vae

    # Optimize for VRAM
    _iclight_t2i_pipe.enable_attention_slicing()
    _iclight_i2i_pipe.enable_attention_slicing()
    _iclight_vae.enable_slicing()
    _iclight_vae.enable_tiling()

    logger.info("IC-Light FBC model loaded successfully")
    return _iclight_t2i_pipe, _iclight_i2i_pipe, _iclight_vae, _iclight_device


# --- Helper functions for IC-Light ---


def _numpy2pytorch(imgs: list[np.ndarray]) -> torch.Tensor:
    """Convert numpy images to pytorch tensor."""
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    h = h.movedim(-1, 1)
    return h


def _pytorch2numpy(imgs: torch.Tensor, quant: bool = True) -> list[np.ndarray]:
    """Convert pytorch tensor to numpy images."""
    results: list[np.ndarray] = []
    for x in imgs:
        y = x.movedim(0, -1)
        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)
        results.append(y)
    return results


def _resize_and_center_crop(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize image and center crop to target dimensions."""
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def _resize_and_pad(
    image: np.ndarray, target_width: int, target_height: int, pad_color: tuple = (127, 127, 127)
) -> np.ndarray:
    """Resize image preserving aspect ratio and pad to target dimensions."""
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    
    scale_factor = min(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    
    resized_image = pil_image.resize(
        (resized_width, resized_height), Image.Resampling.LANCZOS
    )
    
    new_image = Image.new("RGB", (target_width, target_height), pad_color)
    paste_x = (target_width - resized_width) // 2
    paste_y = (target_height - resized_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))
    
    return np.array(new_image)


def _resize_without_crop(
    image: np.ndarray, target_width: int, target_height: int
) -> np.ndarray:
    """Resize image without cropping."""
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )
    return np.array(resized_image)


def _encode_prompt_inner(
    txt: str, tokenizer: Any, text_encoder: Any, device: Any
) -> Any:
    """Encode text prompt to embeddings."""
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x: list[int], p: int, i: int) -> list[int]:
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [
        [id_start] + tokens[i : i + chunk_length] + [id_end]
        for i in range(0, len(tokens), chunk_length)
    ]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state
    return conds


def _encode_prompt_pair(
    positive: str, negative: str, tokenizer: Any, text_encoder: Any, device: Any
) -> tuple[Any, Any]:
    """Encode positive and negative prompts."""
    c = _encode_prompt_inner(positive, tokenizer, text_encoder, device)
    uc = _encode_prompt_inner(negative, tokenizer, text_encoder, device)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


# --- Services ---


class BackgroundRemovalService(AIService):
    """
    Background removal service using RMBG-1.4.

    Strategy: Local model first -> API fallback
    """

    api_url: Optional[str] = None
    use_local_model: bool = True

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.rmbg_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """Remove background from an image. Local first, API fallback."""
        if self.use_local_model:
            try:
                return await self._process_local_model(image_path)
            except Exception as e:
                import traceback

                logger.warning(f"Local RMBG model failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        if self.api_url:
            try:
                return await self._process_api(image_path)
            except Exception as e:
                logger.error(f"API fallback also failed: {e}")
                raise AIServiceError(f"Background removal failed: {e}") from e

        logger.warning("No API configured, using placeholder processing")
        return await self._process_placeholder(image_path)

    async def _process_local_model(self, image_path: str) -> str:
        """Process using local RMBG-1.4 model on GPU."""
        model, device = _load_rmbg_model()

        orig_im = skimage_io.imread(image_path)
        orig_im_size = orig_im.shape[0:2]
        model_input_size = [1024, 1024]

        image_tensor = self._preprocess_image(orig_im, model_input_size)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            result = model(image_tensor)

        mask = self._postprocess_mask(result[0][0], orig_im_size)

        output_path = Path(image_path).parent / "bg_removed.png"
        pil_mask = Image.fromarray(mask)
        orig_image = Image.open(image_path).convert("RGBA")

        no_bg_image = orig_image.copy()
        no_bg_image.putalpha(pil_mask)
        no_bg_image.save(output_path, "PNG")

        logger.info(f"Background removed via local model: {output_path}")
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
            torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
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
        if result.dim() == 3:
            result = result.unsqueeze(0)
        result = F.interpolate(
            result, size=im_size, mode="bilinear", align_corners=False
        )
        result = result.squeeze()
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result - mi) / (ma - mi)
        im_array = (result * 255).cpu().data.numpy().astype(np.uint8)
        return np.asarray(im_array)

    async def _process_api(self, image_path: str) -> str:
        """Process using remote API."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(image_path, "rb") as f:
                image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode("utf-8")
            headers = {"Content-Type": "application/json"}
            settings = get_settings()
            if settings.rmbg_api_key:
                headers["Authorization"] = f"Bearer {settings.rmbg_api_key}"

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json={"image": image_b64},
                headers=headers,
            )
            if response.status_code != 200:
                raise AIServiceError(
                    f"RMBG API error: {response.status_code} - {response.text}"
                )
            result = response.json()
            result_b64 = result.get("result", result.get("image", ""))
            output_path = Path(image_path).parent / "bg_removed.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            return str(output_path)

    async def _process_placeholder(self, image_path: str) -> str:
        """Placeholder: simply converts image to RGBA."""
        output_path = Path(image_path).parent / "bg_removed.png"
        img = Image.open(image_path)
        output_img = img.convert("RGBA") if img.mode != "RGBA" else img
        output_img.save(output_path, "PNG")
        return str(output_path)


class SceneGenerationService(AIService):
    """Scene generation service using Flux or SDXL."""

    api_url: Optional[str] = None

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.flux_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """Generate a scene based on the input image and prompt."""
        prompt = kwargs.get(
            "prompt", "professional product photography, studio lighting"
        )
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
            headers = {"Content-Type": "application/json"}
            settings = get_settings()
            if settings.flux_api_key:
                # Replicate uses "Token <key>"
                if "replicate.com" in str(self.api_url):
                    headers["Authorization"] = f"Token {settings.flux_api_key}"
                else:
                    headers["Authorization"] = f"Bearer {settings.flux_api_key}"

            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json=payload,
                headers=headers,
            )
            if response.status_code != 200:
                raise AIServiceError(
                    f"Scene API error: {response.status_code} - {response.text}"
                )
            result = response.json()
            result_b64 = result.get("result", result.get("image", ""))
            output_path = Path(image_path).parent / "scene.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            return str(output_path)

    @torch.inference_mode()
    async def _process_local(self, image_path: str, prompt: str) -> str:
        """Process using local SD 1.5 pipeline."""
        pipe, device = _load_scene_model()
        
        # Move pipe to GPU before generation
        pipe = pipe.to(device)

        # Typical negative prompt for better quality
        n_prompt = "lowres, bad anatomy, bad quality, worst quality, text, watermark"
        generator = torch.Generator(device=device).manual_seed(42)

        image = pipe(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

        # Move pipe to CPU explicitly and clear cache
        pipe = pipe.to("cpu")
        torch.cuda.empty_cache()

        output_path = Path(image_path).parent / "scene.png"
        image.save(output_path, "PNG")

        logger.info(f"Scene generated via local SD 1.5: {output_path}")
        return str(output_path)


class RelightingService(AIService):
    """
    Relighting service using IC-Light FBC model.

    Strategy: Local model first -> API fallback
    """

    api_url: Optional[str] = None
    use_local_model: bool = True

    def model_post_init(self, __context: object) -> None:
        """Initialize API URL from settings if not provided."""
        if self.api_url is None:
            settings = get_settings()
            self.api_url = settings.iclight_api_url

    async def process(self, image_path: str, **kwargs: str) -> str:
        """
        Apply relighting. Local first, API fallback.

        Args:
            image_path: Path to the foreground image (RGBA).
            background_path: Path to the background image.
            prompt: Text description of desired lighting.
        """
        background_path = kwargs.get("background_path", "")
        prompt = kwargs.get(
            "prompt", "professional product photography, studio lighting"
        )

        if self.use_local_model:
            try:
                return await self._process_local_model(
                    image_path, background_path, prompt
                )
            except Exception as e:
                import traceback

                logger.warning(f"Local IC-Light model failed: {e}")
                logger.warning(f"Full traceback:\n{traceback.format_exc()}")

        if self.api_url:
            try:
                return await self._process_api(image_path, background_path)
            except Exception as e:
                logger.error(f"IC-Light API fallback failed: {e}")
                raise AIServiceError(f"Relighting failed: {e}") from e

        logger.warning("No API configured, using placeholder compositing")
        return await self._process_placeholder(image_path, background_path)

    @torch.inference_mode()
    async def _process_local_model(
        self, fg_path: str, bg_path: str, prompt: str
    ) -> str:
        """
        Process using local IC-Light FBC model on GPU.

        Pipeline: encode fg+bg -> txt2img -> highres img2img -> decode
        """
        t2i_pipe, i2i_pipe, vae, device = _load_iclight_model()

        # Parameters
        image_width, image_height = 512, 512
        steps, cfg = 25, 2.0
        highres_scale, highres_denoise = 1.0, 0.35
        seed = 12345
        a_prompt = "best quality, high detail"
        n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"

        rng = torch.Generator(device=device).manual_seed(seed)

        # Load and resize images
        fg_pil = Image.open(fg_path)
        if fg_pil.mode == "RGBA":
            bg_gray = Image.new("RGB", fg_pil.size, (127, 127, 127))
            bg_gray.paste(fg_pil, mask=fg_pil.split()[3])
            fg_img = np.array(bg_gray)
        else:
            fg_img = np.array(fg_pil.convert("RGB"))
            
        fg = _resize_and_pad(fg_img, image_width, image_height)

        if bg_path and Path(bg_path).exists():
            bg_img = np.array(Image.open(bg_path).convert("RGB"))
            bg = _resize_and_pad(bg_img, image_width, image_height)
        else:
            bg = np.zeros((image_height, image_width, 3), dtype=np.uint8) + 64
            bg_img = bg

        # Encode fg+bg through VAE
        concat_conds = _numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = (
            vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
        )
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        # Encode prompts
        tokenizer = t2i_pipe.tokenizer
        text_encoder = t2i_pipe.text_encoder
        conds, unconds = _encode_prompt_pair(
            prompt + ", " + a_prompt,
            n_prompt,
            tokenizer,
            text_encoder,
            device,
        )

        # First pass: txt2img
        latents = (
            t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )

        torch.cuda.empty_cache()

        # Decode first pass
        pixels = vae.decode(latents).sample
        pixels_np = _pytorch2numpy(pixels)

        # Highres upscale
        hr_w = int(round(image_width * highres_scale / 64.0) * 64)
        hr_h = int(round(image_height * highres_scale / 64.0) * 64)
        pixels_np = [_resize_without_crop(p, hr_w, hr_h) for p in pixels_np]

        torch.cuda.empty_cache()

        # Use PIL image for img2img to avoid range issues and ensure optimization
        hr_pil = Image.fromarray(pixels_np[0])

        # Recalculate hr dimensions and re-encode conditions
        fg_hr = _resize_and_pad(fg_img, hr_w, hr_h)
        bg_hr = _resize_and_pad(bg_img, hr_w, hr_h)
        concat_hr = _numpy2pytorch([fg_hr, bg_hr]).to(
            device=vae.device, dtype=vae.dtype
        )
        concat_hr = (
            vae.encode(concat_hr).latent_dist.mode() * vae.config.scaling_factor
        )
        concat_hr = torch.cat([c[None, ...] for c in concat_hr], dim=1)

        # Second pass: img2img highres
        final_images = i2i_pipe(
            image=hr_pil,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=1,
            generator=rng,
            output_type="np",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_hr},
        ).images

        # Save result
        output_path = Path(fg_path).parent / "relit.png"
        
        # Convert from [0, 1] float32 to [0, 255] uint8
        final_image_np = (final_images[0] * 255).clip(0, 255).astype(np.uint8)
        out_pil = Image.fromarray(final_image_np)

        # Composite original pristine foreground back to preserve sharp text and details
        if fg_pil.mode == "RGBA":
            scale_factor = min(hr_w / fg_pil.width, hr_h / fg_pil.height)
            resized_w = int(round(fg_pil.width * scale_factor))
            resized_h = int(round(fg_pil.height * scale_factor))
            
            resized_fg_pil = fg_pil.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
            
            transparent_layer = Image.new("RGBA", (hr_w, hr_h), (0, 0, 0, 0))
            paste_x = (hr_w - resized_w) // 2
            paste_y = (hr_h - resized_h) // 2
            transparent_layer.paste(resized_fg_pil, (paste_x, paste_y))
            
            out_pil = Image.alpha_composite(out_pil.convert("RGBA"), transparent_layer).convert("RGB")

        out_pil.save(output_path, "PNG")

        logger.info(f"Relighting completed via local IC-Light: {output_path}")
        return str(output_path)

    async def _process_api(self, fg_path: str, bg_path: str) -> str:
        """Process using remote IC-Light API."""
        async with httpx.AsyncClient(timeout=180.0) as client:
            with open(fg_path, "rb") as f:
                fg_data = base64.b64encode(f.read()).decode("utf-8")
            with open(bg_path, "rb") as f:
                bg_data = base64.b64encode(f.read()).decode("utf-8")
            headers = {"Content-Type": "application/json"}
            settings = get_settings()
            if settings.iclight_api_key:
                headers["Authorization"] = f"Bearer {settings.iclight_api_key}"

            payload = {
                "foreground": fg_data,
                "background": bg_data,
                "lighting_mode": "auto",
            }
            response = await client.post(
                self.api_url,  # type: ignore[arg-type]
                json=payload,
                headers=headers,
            )
            if response.status_code != 200:
                raise AIServiceError(
                    f"IC-Light API error: {response.status_code} - {response.text}"
                )
            result = response.json()
            result_b64 = result.get("result", result.get("image", ""))
            output_path = Path(fg_path).parent / "relit.png"
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(result_b64))
            return str(output_path)

    async def _process_placeholder(self, fg_path: str, bg_path: str) -> str:
        """Simple composite placeholder."""
        output_path = Path(fg_path).parent / "relit.png"
        fg_img = Image.open(fg_path)
        if bg_path and Path(bg_path).exists():
            bg_img = Image.open(bg_path).resize(fg_img.size)
            if fg_img.mode == "RGBA":
                result = Image.alpha_composite(bg_img.convert("RGBA"), fg_img)
            else:
                result = fg_img
        else:
            result = fg_img
        result.save(output_path, "PNG")
        return str(output_path)


class AIServiceFactory:
    """Factory for creating AI service instances."""

    @staticmethod
    def get_background_removal_service(
        use_local: bool = True,
    ) -> BackgroundRemovalService:
        """Get background removal service instance."""
        return BackgroundRemovalService(use_local_model=use_local)

    @staticmethod
    def get_scene_generation_service() -> SceneGenerationService:
        """Get scene generation service instance."""
        return SceneGenerationService()

    @staticmethod
    def get_relighting_service(use_local: bool = True) -> RelightingService:
        """Get relighting service instance."""
        return RelightingService(use_local_model=use_local)
