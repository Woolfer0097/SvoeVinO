FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN python -m venv /opt/venv

# Dependencies are installed from a stub package first so that code changes
# do not reinstall torch/transformers on every rebuild. The pip cache mount
# keeps downloaded wheels when pyproject.toml itself changes.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/dinov2_retrieval \
    && touch src/dinov2_retrieval/__init__.py \
    && pip install '.[dev,ml,db,api]'

COPY src ./src
COPY tests ./tests
COPY database ./database
COPY docker-entrypoint.sh /usr/local/bin/dinov2-retrieval-entrypoint.sh

RUN pip install --no-cache-dir --no-deps . \
    && chmod 0555 /usr/local/bin/dinov2-retrieval-entrypoint.sh

ARG APP_UID=1000
ARG APP_GID=1000

# /opt/venv stays root-owned and read-only for the app user: chown-ing it
# would duplicate the multi-gigabyte torch layer.
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /data /home/app/.cache/huggingface \
    && chown -R app:app /app /data /home/app

USER app

ENV DATA_ROOT=/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

ENTRYPOINT ["sh", "/usr/local/bin/dinov2-retrieval-entrypoint.sh"]
CMD ["serve"]
