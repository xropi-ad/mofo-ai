#!/bin/sh
# Build mofo-ai:latest from scratch: no layer cache, fresh base image.
set -e
exec "$(dirname "$0")/build-cached.sh" --no-cache --pull "$@"
