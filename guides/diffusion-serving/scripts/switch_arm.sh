#!/usr/bin/env bash
# Switch the serving stack to one of the benchmark arms.
#   baseline   — plain k8s Service over the pool (no router, no EPP)
#   cost-aware — llm-d EPP (diffusion-cost-scorer / diffusion-load-producer)
#
# cost-aware: helm upgrade with the arm's values file, then a MANDATORY EPP
# rollout restart (the EPP reads its plugin ConfigMap only at startup).
set -euo pipefail
source "$(dirname "$0")/env.sh"

arm="${1:?usage: switch_arm.sh <baseline|cost-aware>}"

case "$arm" in
  baseline)
    kubectl apply -n "$NAMESPACE" -f "$BENCH_DIR/manifests/baseline/service.yaml"
    echo ">>> baseline active: plain k8s Service at $BASELINE_SVC_URL"
    ;;
  cost-aware)
    helm upgrade -i "$HELM_RELEASE" "$ROUTER_STANDALONE_CHART" \
      -f "$LLM_D_DIR/guides/recipes/router/base.values.yaml" \
      -f "$BENCH_DIR/manifests/router/cost-aware.values.yaml" \
      -n "$NAMESPACE" --version "$ROUTER_CHART_VERSION"
    kubectl rollout restart deploy/"$EPP_DEPLOY" -n "$NAMESPACE"
    kubectl rollout status deploy/"$EPP_DEPLOY" -n "$NAMESPACE" --timeout=5m
    echo ">>> cost-aware active: llm-d EPP at $EPP_SVC_URL"
    ;;
  *)
    echo "usage: switch_arm.sh <baseline|cost-aware>" >&2
    exit 1
    ;;
esac
