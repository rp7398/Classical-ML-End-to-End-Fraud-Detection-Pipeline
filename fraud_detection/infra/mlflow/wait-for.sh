#!/usr/bin/env sh
# usage: wait-for.sh host:port timeout_seconds
TARGET=$1
TIMEOUT=${2:-30}
HOST=$(echo "$TARGET" | cut -d: -f1)
PORT=$(echo "$TARGET" | cut -d: -f2)
i=0
while ! nc -z "$HOST" "$PORT" 2>/dev/null; do
  i=$((i+1))
  if [ "$i" -ge "$TIMEOUT" ]; then
    echo "Timed out waiting for $TARGET"
    exit 1
  fi
  sleep 1
done
echo "$TARGET is available"
exit 0
