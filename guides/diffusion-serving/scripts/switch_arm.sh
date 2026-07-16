#!/usr/bin/env bash
# Switch the serving stack to one of the benchmark arms.
#   a — baseline: plain k8s Service over the pool (no router, no EPP)
#   b — cost-aware llm-d EPP (diffusion-cost-scorer / diffusion-load-producer)
#
# Arm b: helm upgrade with the arm's values file, then a MANDATORY EPP rollout
# restart (the EPP reads its plugin ConfigMap only at startup).
set -euo pipefail
source "$(dirname "$0")/env.sh"

arm="${1:?usage: switch_arm.sh <a|b>}"

case "$arm" in
  a)
    kubectl apply -n "$NAMESPACE" -f "$BENCH_DIR/manifests/baseline/service.yaml"
    echo ">>> arm a active: plain k8s Service at $BASELINE_SVC_URL"
    ;;
  b)
    helm upgrade -i "$HELM_RELEASE" "$ROUTER_STANDALONE_CHART" \
      -f "$LLM_D_DIR/guides/recipes/router/base.values.yaml" \
      -f "$BENCH_DIR/manifests/router/arm-b.values.yaml" \
      -n "$NAMESPACE" --version "$ROUTER_CHART_VERSION"
    kubectl rollout restart deploy/"$EPP_DEPLOY" -n "$NAMESPACE"
    kubectl rollout status deploy/"$EPP_DEPLOY" -n "$NAMESPACE" --timeout=5m
    echo ">>> arm b active: cost-aware EPP at $EPP_SVC_URL"
    ;;
  *)
    echo "usage: switch_arm.sh <a|b>" >&2
    exit 1
    ;;
esac
