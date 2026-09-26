FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/home/app/.cache/torch

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

# Dependencies are installed from a stub package first so that code changes
# do not reinstall torch or LightGlue on every rebuild.
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/superpoint \
    && touch src/superpoint/__init__.py \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
        'torch==2.4.1' 'torchvision==0.19.1' \
    && printf '%s\n' 'torch==2.4.1' 'torchvision==0.19.1' > /tmp/torch-constraints.txt \
    && pip install -c /tmp/torch-constraints.txt '.[dev,ml]' \
    && (pip uninstall -y opencv-python || true) \
    && pip install --force-reinstall 'opencv-python-headless>=4.8,<5.0' \
    && python -c "import importlib.metadata as m, cv2, lightglue.utils, torch; names={d.metadata['Name'].lower() for d in m.distributions()}; assert 'opencv-python' not in names; print(torch.__version__, cv2.__version__)"

COPY src ./src
COPY tests ./tests
COPY docker-entrypoint.sh /usr/local/bin/superpoint-entrypoint.sh

RUN pip install --no-cache-dir --no-deps . \
    && chmod 0555 /usr/local/bin/superpoint-entrypoint.sh

ARG APP_UID=1000
ARG APP_GID=1000

# /opt/venv stays root-owned and read-only for the app user: chown-ing it
# would duplicate the multi-gigabyte torch layer.
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" \
        --create-home --home-dir /home/app --shell /usr/sbin/nologin app \
    && mkdir -p /data /home/app/.cache/torch \
    && chown -R app:app /app /data /home/app

USER app

ENV DATA_ROOT=/data \
    DEVICE=auto

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import superpoint; from superpoint.config import get_data_root; get_data_root()"

ENTRYPOINT ["sh", "/usr/local/bin/superpoint-entrypoint.sh"]
CMD ["--help"]
