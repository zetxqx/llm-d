#!/usr/bin/env bash
# Pin the model pods to stable shard identities for calibration:
# label pod[i] (sorted by name) with bench.costaware/shard=i, so
# calibrate.sh can measure service times against one pod (shard-0) directly.
# Re-run after any pod replacement (spot preemption).
set -euo pipefail
source "$(dirname "$0")/env.sh"

mapfile -t pods < <(kubectl get pods -n "$NAMESPACE" -l "$POOL_SELECTOR" \
  --field-selector=status.phase=Running -o name | sort)

if [[ ${#pods[@]} -ne "$REPLICAS" ]]; then
  echo "ERROR: expected exactly $REPLICAS running model pods, found ${#pods[@]}" >&2
  exit 1
fi

for i in "${!pods[@]}"; do
  kubectl label -n "$NAMESPACE" "${pods[$i]}" "bench.costaware/shard=$i" --overwrite
done

# Assert the shard-0 Service (used by calibrate.sh) resolves to one endpoint.
n=$(kubectl get endpointslices -n "$NAMESPACE" \
      -l "kubernetes.io/service-name=qwen-decode-shard-0" \
      -o jsonpath='{range .items[*].endpoints[*]}{.addresses[0]}{"\n"}{end}' | grep -c . || true)
if [[ "$n" != "1" ]]; then
  echo "ERROR: qwen-decode-shard-0 has $n endpoints (want 1). Is shard-services.yaml applied?" >&2
  exit 1
fi
echo ">>> ${#pods[@]} shards labeled; shard-0=${pods[0]#pod/}"
