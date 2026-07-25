FROM python:3.13-slim

# ── Install uv ─────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Install system deps ────────────────────────────────────────
RUN apt-get update -qq && apt-get install -y -qq \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python deps ────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml uv.lock ./
# sentence-transformers[onnx] uses ONNX Runtime (~50 MB) instead of
# PyTorch (~526 MB CUDA wheel). No torch = no nvidia-* / CUDA bloat.
RUN uv sync --frozen --no-dev

# ── Copy source ────────────────────────────────────────────────
COPY app/ ./app/
COPY main.py ./
COPY conftest.py ./

# ── Pre-download embedding model (D-018) ───────────────────────
# fastembed downloads model to ~/.cache/fastembed on first call.
# Running it at build time ensures zero latency on first request.
RUN uv run python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(); \
print('Model downloaded')"

# ── Expose + start ─────────────────────────────────────────────
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
