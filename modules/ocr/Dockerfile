FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OCR_OUTPUT_DIR=/app/outputs \
    OCR_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

COPY pyproject.toml README.md ./
COPY src ./src

# The OCR container runs on CPU. Install the CPU wheel before resolving the
# matching extra so pip does not pull CUDA libraries into this image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir 'torch==2.8.0' \
    --index-url "${TORCH_INDEX_URL}"
RUN pip install --no-cache-dir '.[matching]'

ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /app/outputs \
    && chown -R app:app /app

RUN mkdir -p /home/app/.cache/huggingface /home/app/.paddlex /home/app/.paddleocr \
    && chown -R app:app /home/app/.cache /home/app/.paddlex /home/app/.paddleocr

USER app

EXPOSE 8000

CMD ["uvicorn", "wine_ocr.entrypoints.rest:app", "--host", "0.0.0.0", "--port", "8000"]
