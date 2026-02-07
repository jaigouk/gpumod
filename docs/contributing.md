---
title: Contributing to gpumod
description: Development setup, running tests, code quality checks with ruff and mypy, and pull request guidelines for gpumod.
---

# Contributing

Contributions are welcome. Please follow these guidelines:

## Development setup

```bash
git clone https://github.com/jaigouk/gpumod.git
cd gpumod
uv sync   # installs all dependencies including dev group
```

## Running tests

```bash
# Full test suite
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ -v --cov=src/gpumod --cov-fail-under=80

# Run only unit tests (skip integration and e2e)
uv run pytest tests/ -v -m "not integration"

# Run E2E tests on GPU machines
uv run pytest tests/e2e/ -v

# CPU-only CI (skip GPU/Docker tests)
uv run pytest tests/ -v -m "not gpu_required and not docker_required"
```

## Code quality

gpumod enforces strict code quality via ruff, mypy, and pytest:

```bash
# Lint
uv run ruff check src/ tests/

# Format check
uv run ruff format --check src/ tests/

# Type check (strict mode)
uv run mypy src/ --strict

# Full quality gate
uv run ruff check src/ tests/ && \
  uv run ruff format --check src/ tests/ && \
  uv run mypy src/ --strict && \
  uv run pytest tests/ -v --cov=src/gpumod --cov-fail-under=80
```

## Code style

- Python >= 3.12 with `from __future__ import annotations`
- Async-first: use `async`/`await` for I/O operations
- Pydantic v2 models with `ConfigDict(extra="forbid")`
- Type annotations on all functions (mypy strict mode)
- Docstrings in NumPy/Google style
- Import sorting via ruff (isort rules)
- Line length: 99 characters

## Pull requests

1. Create a feature branch from `main`.
2. Write tests for new functionality.
3. Ensure the full quality gate passes.
4. Submit a pull request with a clear description.
