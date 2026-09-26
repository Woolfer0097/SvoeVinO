#!/bin/sh
set -eu

if [ "${1:-}" = "pytest" ]; then
    shift
    exec pytest "$@"
fi

if [ "${1:-}" = "serve" ]; then
    shift
    exec uvicorn --factory dinov2_retrieval.entrypoints.api:create_app \
        --host 0.0.0.0 --port "${PORT:-8000}" "$@"
fi

exec python -m dinov2_retrieval.cli "$@"
