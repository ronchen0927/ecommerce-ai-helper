# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run API server
uv run uvicorn app.main:app --reload

# Run Celery worker (Linux/macOS)
uv run celery -A app.core.celery_app worker --loglevel=info

# Run Celery worker (Windows)
uv run celery -A app.core.celery_app worker --pool=solo --loglevel=info

# Run tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_api.py -v

# Run a single test
uv run pytest tests/test_api.py::test_health_check -v

# Lint
uv run ruff check .

# Type check
uv run mypy .
```

## Architecture

**Request flow:** Client → FastAPI (`POST /api/v1/upload`) → Redis → Celery worker → AI service → Storage → Client polls `GET /api/v1/task-status/{id}`.

**Two processing modes** selected at upload time via the `mode` form field:
- `remove_bg` — RMBG-1.4 background removal → outputs transparent PNG
- `edit` — FireRed-Image-Edit-1.1 instruction-based editing → outputs PNG

**AI service strategy (both modes):** Replicate API → Custom cloud API → local GPU model. Configured via env vars; if no API token is set, falls back to local inference automatically.

### Key files

| File | Role |
|---|---|
| `app/core/config.py` | All settings via `pydantic-settings`; `get_settings()` is `@lru_cache` — call `get_settings.cache_clear()` in tests that mutate settings |
| `app/api/routes.py` | FastAPI endpoints for image processing; `task_metadata_store` is an **in-memory dict** (task metadata only — actual task STATUS comes from Celery/Redis via `AsyncResult`) |
| `app/api/video_routes.py` | FastAPI endpoints for video generation; `video_task_store` is an **in-memory dict** mirroring the same pattern as `task_metadata_store` |
| `app/tasks/image_processing.py` | Celery task `process_image`; bridges async AI services to sync Celery via `run_async()` helper; uses a temp dir per task that is always cleaned up |
| `app/tasks/video_processing.py` | Celery task `process_video`; generates clips **sequentially** then concatenates with FFmpeg; exceptions are caught and returned as `{"status": "FAILED"}` dicts so Celery records `SUCCESS` state — status endpoint must check `result["status"]` |
| `app/services/ai_service.py` | `BackgroundRemovalService` and `FireRedEditService` (both extend `AIService`); AI models are **lazy-loaded** into module-level globals (`_rmbg_model`, `_firered_pipe`) on first use |
| `app/services/storyboard_service.py` | Calls GPT-5.4-mini Vision to generate a shot-by-shot storyboard; falls back to a built-in template if the OpenAI call fails or returns invalid JSON |
| `app/services/video_service.py` | `generate_clip()` — uploads image to Replicate, polls until done, downloads MP4; retries up to 3× on 429 with 10 / 30 / 90 s backoff. `extract_last_frame()` — extracts last frame of a clip via `ffmpeg -sseof` for next-clip conditioning. `concatenate_clips()` — xfade crossfade with hard-cut fallback |
| `app/services/storage.py` | `StorageService` interface with `LocalStorage` and `GCSStorage` implementations |
| `app/core/auth.py` | API key + JWT auth and rate limiting; both are **feature-flagged off by default** (`AUTH_ENABLED=false`, `RATE_LIMIT_ENABLED=false`) |

### Notable implementation details

- **RMBG-1.4 transformer 5.x patch** (`ai_service.py:_load_rmbg_model`): `BriaRMBG` (loaded with `trust_remote_code`) is missing `all_tied_weights_keys` required by transformers 5.x. The loader temporarily monkey-patches `get_class_from_dynamic_module` to inject the missing property before restoring the original.

- **FireRed image dimensions**: Input images must be divisible by 64 for diffusers. The local inference path (`_process_local_model`) and Replicate path both resize and clamp dimensions accordingly.

- **FireRed GGUF vs standard**: If `FIRERED_MODEL_PATH` points to a `.gguf` file, the local loader uses `GGUFQuantizationConfig` + `QwenImageTransformer2DModel`; otherwise it loads the standard HuggingFace model.

- **Task metadata vs task status**: `task_metadata_store` in `routes.py` holds upload metadata (created_at, original URL, mode). Task status (PENDING/EDITING/COMPLETED/FAILED) is stored in Celery/Redis and retrieved via `AsyncResult`. The two are joined in `get_task_status` and `get_result`. `video_task_store` in `video_routes.py` follows the same pattern for video tasks.

- **Video task failure encoding**: `process_video` never raises — it catches all exceptions and returns `{"status": "FAILED", "error": ...}`. This means Celery always records `SUCCESS`. The status endpoint must inspect `result["status"]` in addition to `AsyncResult.state` to distinguish a real completion from a silent failure.

- **Sequential clip generation**: Clips are generated one at a time (not `asyncio.gather`) to avoid Replicate concurrent request limits. Progress is updated after every clip via `task.update_progress()`.

- **Clip transition continuity (i2v chaining)**: After each successful clip, `extract_last_frame()` saves the last video frame as a PNG; this frame is passed as Wan's `image` (first frame) for the next clip so each scene starts exactly where the previous one ended. The original product image is only used for the first clip; subsequent clips chain from the previous clip's last frame. On clip failure, `last_frame_path` is reset to `None` so the next clip falls back to the product image. `concatenate_clips()` uses FFmpeg `xfade=transition=fade` with `_XFADE_DURATION = 0.5s`; xfade offset per transition = cumulative sum of preceding clip durations minus `n × fade_duration`. Falls back to hard-cut concat if xfade fails (e.g. codec mismatch).

- **Wan i2v motion-focused prompts**: `storyboard_service.py` prompts describe **how things move**, not what they look like — the input image already defines all visual content. Prompt structure: `[Subject motion] + [Camera movement] + [Subtle natural effect (optional)] + [Speed modifier]`. Allowed natural effects: highlight glides across a surface, shadow shifts as angle changes, reflections respond to camera angle, shallow depth of field softens slightly. Forbidden: floating particles, magical mist, pulsing bokeh, or any fantasy effects. If you modify `_SYSTEM_PROMPT`, preserve this constraint: never describe color, shape, or appearance in the video prompt.

### Configuration

Copy `.env.example` to `.env`. Key variables:

```env
STORAGE_TYPE=local            # "local" or "gcs"
REDIS_URL=redis://localhost:6379/0
REPLICATE_API_TOKEN=          # Recommended for production (image processing)
RMBG_API_URL=                 # Custom RMBG API (alternative to Replicate)
FIRERED_API_URL=              # Custom FireRed API (alternative to Replicate)
FIRERED_MODEL_PATH=           # Path to local .gguf file for local inference
AUTH_ENABLED=false
RATE_LIMIT_ENABLED=false

# Video generation
OPENAI_API_KEY=               # GPT-5.4-mini Vision for storyboard generation
REPLICATE_VIDEO_MODEL=wan-video/wan-2.2-i2v-fast   # Replicate model for clip generation
```

### CI/CD

GitHub Actions (`.github/workflows/ci.yml`): **Lint & type check → Unit tests → Docker build** on every push/PR to `main`.

Production deploy via Google Cloud Build → Cloud Run:
```bash
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION=asia-east1,_SERVICE_NAME=ai-ecommerce-media-studio
```

**Two cloud workloads, one image.** `cloudbuild.yaml` deploys both:
- The **API** as a Cloud Run *service* (`uvicorn`, scale-to-zero) — Step 3.
- The **Celery worker** as a Cloud Run *WorkerPool* (`deploy/cloudrun-worker.yaml`) — Step 4. It overrides the image's `uvicorn` CMD with `celery ... worker`. A worker pool (not a regular service) is required because the worker only polls Redis and never listens on `$PORT`, so it would fail a service health check; worker pools are always-CPU-allocated pull consumers. **Without the worker, dispatched tasks sit in Redis and never leave PENDING.** The production image has torch/diffusers stripped, so the worker only runs the Replicate path (local-model fallback would crash — fine, since `REPLICATE_API_TOKEN` is always set in cloud). Worker pools don't autoscale: instance count is fixed via `run.googleapis.com/manualInstanceCount` / `--instances`. The worker needs GCS write + Secret Manager access on its service account; it does **not** need `OPENAI_API_KEY` (storyboard runs in the API route) or `AUTH_*`/`API_KEYS`.
