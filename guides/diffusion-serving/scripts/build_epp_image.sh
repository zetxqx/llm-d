#!/usr/bin/env bash
# Build + push the EPP image that contains the diffusion-cost-scorer and
# diffusion-load-producer, from the local llm-d-inference-scheduler checkout
# (branch feat/diffusion-declared-cost).
#
# IMPORTANT: the deployed images-gen-v2 tag predates the cost plugins — the cost-aware arm
# does not work without this build. The Makefile builds from the working tree,
# so uncommitted changes ship too; commit first for a traceable image.
set -euo pipefail
source "$(dirname "$0")/env.sh"

: "${SCHED_DIR:?set SCHED_DIR to a llm-d-inference-scheduler checkout on branch feat/diffusion-declared-cost}"
cd "$SCHED_DIR"

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "feat/diffusion-declared-cost" ]]; then
  echo "ERROR: llm-d-inference-scheduler is on '$branch', expected feat/diffusion-declared-cost" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "WARNING: working tree is dirty — the image will include uncommitted changes." >&2
  echo "         Commit first for a traceable COMMIT_SHA (Ctrl-C to abort, Enter to continue)." >&2
  read -r
fi

echo ">>> building ${EPP_REGISTRY}/llm-d-router-endpoint-picker:${EPP_TAG}"
make image-build-epp image-push-epp \
  IMAGE_REGISTRY="$EPP_REGISTRY" \
  EPP_TAG="$EPP_TAG" \
  TARGETARCH=amd64

echo ">>> pushed ${EPP_REGISTRY}/llm-d-router-endpoint-picker:${EPP_TAG}"
