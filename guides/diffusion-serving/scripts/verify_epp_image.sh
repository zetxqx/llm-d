#!/usr/bin/env bash
# Prove the DEPLOYED EPP actually contains the diffusion cost plugins.
#
# Two probes:
#   1. Config probe — if the EPP was started with the arm-b plugin config but
#      the binary lacks the plugin types, it crash-loops with an
#      unknown-plugin-type error at startup.
#   2. Log probe — the scorer logs its per-endpoint cost units at default
#      verbosity on every scheduling decision (scorer.go log line), so any
#      routed request through arm B must produce matching log lines.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo ">>> EPP image currently deployed:"
kubectl get deploy "$EPP_DEPLOY" -n "$NAMESPACE" \
  -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'

echo ">>> EPP pod status (CrashLoopBackOff here after switching to arm B means the image lacks the plugins):"
kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=epp -o wide 2>/dev/null \
  || kubectl get pods -n "$NAMESPACE" | grep -i epp

echo ">>> recent EPP logs mentioning diffusion cost (empty until traffic flows through arm B):"
kubectl logs deploy/"$EPP_DEPLOY" -n "$NAMESPACE" --tail=2000 2>/dev/null \
  | grep -iE "diffusion|cost unit" | tail -20 || echo "(no diffusion cost lines found)"
