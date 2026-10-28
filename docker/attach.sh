#!/bin/sh
# Open another shell in the newest running mofo-ai container (started by run.py).
exec docker exec -it "$(docker ps -q --latest --filter ancestor=mofo-ai:latest)" bash
