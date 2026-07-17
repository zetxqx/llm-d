#!/usr/bin/env bash
# Sample KV metrics at ~2Hz -> CSV: epoch,token_usage,kv_available,kv_evictable,evicted_total
OUT="$1"; POOL=143872
echo "t,token_usage,kv_available,kv_evictable,evicted_total" > "$OUT"
while true; do
  m=$(curl -s http://127.0.0.1:30000/metrics 2>/dev/null) || true
  tu=$(echo "$m" | awk -F' ' '/^sglang:token_usage{/{print $2}')
  av=$(echo "$m" | awk -F' ' '/^sglang:kv_available_tokens{/{print $2}')
  ev=$(echo "$m" | awk -F' ' '/^sglang:kv_evictable_tokens{/{print $2}')
  fe=$(echo "$m" | awk -F' ' '/^sglang:evicted_tokens_total{.*RadixCache/{print $2}')
  echo "$(date +%s.%N),${tu:-NA},${av:-NA},${ev:-NA},${fe:-0}" >> "$OUT"
  sleep 0.5
done
