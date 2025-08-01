#!/usr/bin/env bash
# for debugging
# catch errors early and pick up the last error that occurred before exiting
set -eo pipefail

# Default to port 8080 if not set.
PORT=${PORT:-8080}

exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 "src.main:create_app()"