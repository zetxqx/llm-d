#!/usr/bin/env bash
# Smoke test for the vLLM-Omni services running in the `omni` namespace.
#
#   vllm-omni-image  -> Z-Image-Turbo  (diffusion: text -> image)
#
# (The Qwen2.5-Omni-3B chat service this script also tested was removed
# 2026-07-15.) ClusterIP, so we port-forward to localhost first.
#
# Usage:
#   ./smoke-test.sh image    # test the diffusion image model (default)
set -euo pipefail

NS="${NS:-omni}"
IMAGE_LOCAL_PORT="${IMAGE_LOCAL_PORT:-8001}"
WHICH="${1:-image}"

PF_PIDS=()
cleanup() { for p in "${PF_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

port_forward() {  # svc local_port
  local svc="$1" lport="$2"
  echo ">> port-forward svc/$svc $lport:8000 (ns=$NS)"
  kubectl port-forward -n "$NS" "svc/$svc" "$lport:8000" >/dev/null 2>&1 &
  PF_PIDS+=("$!")
  # wait for the tunnel to come up
  for _ in $(seq 1 20); do
    curl -sf "http://localhost:$lport/health" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "!! $svc did not become healthy on localhost:$lport"; return 1
}

test_image() {
  local base="http://localhost:$IMAGE_LOCAL_PORT"
  echo "== health ==";  curl -sf "$base/health" && echo " OK"
  echo "== models =="; curl -s "$base/v1/models" | jq -r '.data[].id'
  echo "== image generation -> out2.png =="
  curl -s "$base/v1/images/generations" \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "Tongyi-MAI/Z-Image-Turbo",
      "prompt": "a dragon over the Green Mountains of Vermont, golden hour",
      "size": "1024x1024",
      "seed": 42
    }' | jq -r '.data[0].b64_json' | base64 -d > out2.png
  echo "wrote ou2.png ($(wc -c < out2.png) bytes)"
}

[[ "$WHICH" == "image" || "$WHICH" == "all" ]] && { port_forward vllm-omni-image "$IMAGE_LOCAL_PORT"; test_image; }
echo "done."
