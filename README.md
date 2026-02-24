# Ecommerce Visual Pro

AI-powered e-commerce product photo optimization microservice.

## Features

- **Background Removal** — RMBG-1.4 (local GPU first, API fallback)
- **Scene Generation** — Flux/SDXL for custom backgrounds
- **Relighting** — IC-Light for lighting harmonization
- **Async Processing** — Celery + Redis task queue
- **Storage** — Local filesystem with GCS-ready interface

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA (RTX 4070 recommended, 12GB VRAM)
- Redis (for task queue)

## Quick Start

```bash
# Install dependencies
uv sync

# Run development server
uv run uvicorn app.main:app --reload

# Run Celery worker (requires Redis)
uv run celery -A app.core.celery_app worker --loglevel=info
```

## Example Prompts

Here are some effective prompts for the `scene_prompt` parameter to get the best results for e-commerce products:

### Minimal & Clean (Best for Tech/Gadgets)

- `professional product photography, clean white studio background, soft studio lighting, sharp focus, 8k resolution, photorealistic`
- `product resting on a sleek black marble podium, dark studio styling, dramatic rim lighting, premium aesthetic, highly detailed`

### Lifestyle & Contextual (Best for Fashion/Home)

- `product placed on a cozy wooden table, blurred bright cafe background in the morning, soft warm sunlight filtering through a window, shallow depth of field`
- `skincare bottle on a natural stone block, surrounded by subtle green palm shadows, bright airy bathroom setting, spa atmosphere, photorealistic`

### Creative & Vibrant (Best for Cosmetics/Beverages)

- `product floating in crystal clear splashing water, bright summer lighting, turquoise background, high speed photography, refreshing vibe`
- `product surrounded by floating pastel geometric shapes, vibrant studio lighting, pop art style, clean colorful background, 4k`

> **Pro Tip**: Always append modifiers like `professional product photography`, `studio lighting`, or `photorealistic` to steer the model toward commercial quality.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/upload` | Upload image for processing |
| GET | `/api/v1/task-status/{id}` | Get task status |
| GET | `/api/v1/result/{id}` | Get processing result |

## Project Structure

```text
app/
├── api/routes.py       # FastAPI endpoints
├── core/
│   ├── config.py       # Settings (pydantic-settings)
│   └── celery_app.py   # Celery configuration
├── schemas/task.py     # Pydantic models
├── services/
│   ├── ai_service.py   # AI model integrations
│   └── storage.py      # Storage service
└── tasks/              # Celery tasks
tests/                  # pytest test suite
```

## Configuration

Create `.env` file:

```env
# Storage
STORAGE_TYPE=local
LOCAL_STORAGE_PATH=./storage

# Redis
REDIS_URL=redis://localhost:6379/0

# AI APIs (optional, local model used by default)
RMBG_API_URL=
FLUX_API_URL=
ICLIGHT_API_URL=
```

## Development

```bash
# Run tests
uv run pytest -v

# Lint
uv run ruff check .

# Type check
uv run mypy .
```

## License

MIT
