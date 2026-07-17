#!/usr/bin/env bash
# Sample KV metrics at ~1Hz from one SGLang worker -> CSV.
# Usage: ./sample_metrics_ab.sh <base_url> <out.csv>
# Adapted from the gist's sample_metrics.sh for the GKE A/B run
# (parameterized URL, pool=321614 on H100 TP2 Qwen3-32B).
URL="$1"; OUT="$2"
echo "t,token_usage,kv_available,kv_evictable,evicted_total" > "$OUT"
while true; do
  m=$(curl -s -m 5 "$URL/metrics" 2>/dev/null) || true
  tu=$(echo "$m" | awk '/^sglang:token_usage{/{print $2}')
  av=$(echo "$m" | awk '/^sglang:kv_available_tokens{/{print $2}')
  ev=$(echo "$m" | awk '/^sglang:kv_evictable_tokens{/{print $2}')
  fe=$(echo "$m" | awk '/^sglang:evicted_tokens_total{.*RadixCache/{print $2}')
  echo "$(date +%s),${tu:-NA},${av:-NA},${ev:-NA},${fe:-0}" >> "$OUT"
  sleep 1
done
