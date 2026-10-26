#!/bin/sh
# Build mofo-ai:latest, reusing the docker layer cache.
set -e
cd "$(dirname "$0")"
docker build \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    -t mofo-ai:latest \
    "$@" .
