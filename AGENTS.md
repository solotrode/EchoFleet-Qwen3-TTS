# AGENTS.md - EchoFleet Qwen3-TTS

Quick reference for AI agents working on this codebase.

## Build & Run

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Docker build
docker build -t echofleet-qwen3-tts .
docker run --gpus all -p 8000:8000 echofleet-qwen3-tts

# Run API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Lint & Format

```bash
# Format code (required before commit)
black --line-length 100 .
isort --profile black .

# Type checking
mypy .

# Linting
flake8 .

# All checks
black --line-length 100 . && isort --profile black . && mypy . && flake8 .
```

## Testing

```bash
# Run all tests
pytest

# Run single test file
pytest tests/unit/test_models.py

# Run single test
pytest tests/unit/test_models.py::TestModelManager::test_init_creates_model_manager

# Run with coverage
pytest --cov=. --cov-report=term --cov-fail-under=80
```

## Code Style

### Imports (alphabetical, 3 groups)
```python
# 1. Standard library
import os
from typing import Dict, List, Optional

# 2. Third-party
import torch
from fastapi import FastAPI

# 3. Local
from models.loader import ModelManager
```

### Type Hints (required)
```python
def process_text(text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    pass
```

### Naming
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_single_underscore`

### Docstrings (Google style)
```python
def generate(text: str) -> torch.Tensor:
    """Generate audio from text.
    
    Args:
        text: Input text to synthesize.
    
    Returns:
        Audio tensor of shape (samples,).
    
    Raises:
        ValueError: If text is empty.
    """
    pass
```

### Error Handling
- Use custom exceptions inheriting from `Qwen3TTSError`
- Never use bare `except:`
- Always log errors before raising
- Use `raise ... from e` for exception chaining

### Logging
```python
from utils.logging import get_logger
logger = get_logger(__name__)

logger.info("Model loaded", extra={"model": "base", "duration_ms": 150})
logger.error("Failed to load", extra={"error": str(e)})
```

## Project Structure

```
├── api/           # FastAPI routes, schemas, middleware
├── config/        # Settings (Pydantic)
├── inference/     # TTS inference engines
├── models/        # Model loading & management
├── utils/         # Audio processing, GPU utils, logging
├── tests/         # Unit/integration tests
└── scripts/       # Utility scripts
```

## Key Patterns

### GPU Memory Cleanup (always use between model switches)
```python
def aggressive_vram_cleanup():
    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect()
```

### Config (use environment variables)
```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    model_cache_dir: str = Field(default="/models", env="MODEL_CACHE_DIR")
    api_port: int = Field(default=8000, env="API_PORT")
```

## Pre-commit Checklist

- [ ] `black --line-length 100 .`
- [ ] `isort --profile black .`
- [ ] `mypy .` (no errors)
- [ ] `pytest` (all passing)
- [ ] Type hints on all functions
- [ ] Google-style docstrings on public functions
- [ ] No `print()` statements (use logger)
- [ ] No hardcoded config values
