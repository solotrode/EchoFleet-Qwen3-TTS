#!/bin/sh
set -eu

echo "=== BEFORE: tts keys and contents ==="
redis-cli --scan --pattern 'tts:jobs*' | sed 's/^/KEY: /' || true
for k in $(redis-cli --raw --scan --pattern 'tts:jobs*' || true); do
  echo "LIST: $k"
  redis-cli LRANGE "$k" 0 -1 || true
done

echo "=== BEFORE: job payloads ==="
for p in $(redis-cli --raw --scan --pattern 'tts:job:*:payload' || true); do
  echo "PAYLOAD_KEY: $p"
  redis-cli GET "$p" || true
done

echo "=== BEFORE: job hashes ==="
for h in $(redis-cli --raw --scan --pattern 'tts:job:*' || true); do
  typ=$(redis-cli TYPE "$h")
  # skip payload keys
  case "$h" in
    *:payload) continue ;;
  esac
  if [ "$typ" = "hash" ]; then
    echo "HASH: $h"
    redis-cli HGETALL "$h" || true
  fi
done

echo
echo "=== ACTION: marking job hashes as killed and deleting queues ==="
for h in $(redis-cli --raw --scan --pattern 'tts:job:*' || true); do
  case "$h" in
    *:payload) continue ;;
  esac
  if [ "$(redis-cli TYPE "$h")" = "hash" ]; then
    redis-cli HSET "$h" status killed || true
    echo "MARKED_KILLED: $h"
  fi
done

for k in $(redis-cli --raw --scan --pattern 'tts:jobs*' || true); do
  redis-cli DEL "$k" || true
  echo "DELETED: $k"
done

echo "=== DONE ==="
