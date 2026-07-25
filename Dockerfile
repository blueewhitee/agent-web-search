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
