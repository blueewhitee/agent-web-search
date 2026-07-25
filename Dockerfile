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
# Install CPU-only PyTorch first to avoid pulling ~2GB of CUDA libs
# (sentence-transformers depends on torch; without this, uv pulls the
# full CUDA package that only matters for GPU training, not inference).
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu
RUN uv sync --frozen --no-dev

# ── Copy source ────────────────────────────────────────────────
COPY app/ ./app/
COPY main.py ./
COPY conftest.py ./

# ── Pre-download embedding model (D-018) ───────────────────────
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Model downloaded')"

# ── Expose + start ─────────────────────────────────────────────
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
