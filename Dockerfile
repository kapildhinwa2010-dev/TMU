FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY tests ./tests
COPY docs ./docs
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --upgrade pip && pip install -e .

CMD ["python", "-m", "tmu.cli", "--mode", "advanced", "--epochs", "1", "--batch-size", "2", "--seq-len", "32", "--samples", "64", "--max-gen-tokens", "16"]