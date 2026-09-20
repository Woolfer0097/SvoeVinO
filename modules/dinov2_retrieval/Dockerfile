FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN python -m venv /opt/venv

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY docker-entrypoint.sh /usr/local/bin/dinov2-retrieval-entrypoint.sh

RUN pip install --no-cache-dir '.[dev,ml]' \
    && chmod 0555 /usr/local/bin/dinov2-retrieval-entrypoint.sh

ARG APP_UID=1000
ARG APP_GID=1000

RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /data /root/.cache/huggingface \
    && chown -R app:app /app /opt/venv /data /root/.cache \
    && chmod 0755 /root /root/.cache /root/.cache/huggingface

USER app

ENV DATA_ROOT=/data

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import dinov2_retrieval; from dinov2_retrieval.config import get_data_root; get_data_root()"

ENTRYPOINT ["sh", "/usr/local/bin/dinov2-retrieval-entrypoint.sh"]
CMD ["--help"]
