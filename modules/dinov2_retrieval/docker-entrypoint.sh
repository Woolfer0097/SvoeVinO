#!/bin/sh
set -eu

if [ "${1:-}" = "pytest" ]; then
    shift
    exec pytest "$@"
fi

exec python -m dinov2_retrieval.cli "$@"
