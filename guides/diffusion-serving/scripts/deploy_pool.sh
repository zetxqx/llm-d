#!/usr/bin/env bash
# Deploy the $REPLICAS-replica Qwen-Image pool with --log-stats and wait for
# readiness.
set -euo pipefail
source "$(dirname "$0")/env.sh"

kubectl apply -n "$NAMESPACE" -k "$BENCH_DIR/manifests/modelserver"

echo ">>> waiting for $REPLICAS/$REPLICAS model server replicas (first start downloads ~57 GB of weights; be patient)"
kubectl rollout status deploy/optimized-baseline-omni-qwen-image-nvidia-gpu-vllm-decode \
  -n "$NAMESPACE" --timeout=45m

echo ">>> verifying the vllm_omni gauges are exported (requires --log-stats)"
pod=$(kubectl get pods -n "$NAMESPACE" -l "$POOL_SELECTOR" -o jsonpath='{.items[0].metadata.name}')
if kubectl exec -n "$NAMESPACE" "$pod" -c modelserver -- \
    python -c "import urllib.request; body = urllib.request.urlopen('http://localhost:8000/metrics', timeout=5).read().decode(); assert 'vllm_omni:num_requests_running' in body, 'gauge missing'"; then
  echo ">>> OK: vllm_omni:num_requests_running present on $pod"
else
  echo "ERROR: vllm_omni:num_requests_running missing — is --log-stats in the args?" >&2
  exit 1
fi
