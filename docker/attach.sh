#!/bin/sh
# Open another shell in the running mofo-ai container (started by run.py).
exec docker exec -it mofo-ai bash
