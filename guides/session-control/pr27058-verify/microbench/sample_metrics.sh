#!/bin/sh
# Shared pod-side KV metrics sampler — runs INSIDE an sglang pod, prints a
# 1 Hz CSV to stdout. Stream it out of the cluster with:
#
#   kubectl -n $NS cp sample_metrics.sh <pod>:/tmp/sample_metrics.sh
#   kubectl -n $NS exec <pod> -- sh /tmp/sample_metrics.sh > pod_metrics.csv &
#
# Columns: t (epoch s), token_usage (active fraction of pool), kv_available
# (free tokens), kv_evictable (resident cache tokens), evicted_total
# (cumulative forced-eviction tokens).
echo "t,token_usage,kv_available,kv_evictable,evicted_total"
while true; do
  m=$(curl -s -m 5 localhost:8000/metrics 2>/dev/null)
  tu=$(echo "$m" | awk '/^sglang:token_usage{/{print $2}')
  av=$(echo "$m" | awk '/^sglang:kv_available_tokens{/{print $2}')
  ev=$(echo "$m" | awk '/^sglang:kv_evictable_tokens{/{print $2}')
  fe=$(echo "$m" | awk '/^sglang:evicted_tokens_total{/{s+=$2} END{print s+0}')
  echo "$(date +%s),${tu:-NA},${av:-NA},${ev:-NA},${fe:-0}"
  sleep 1
done
