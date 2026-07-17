#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch a SINGLE raw SGLang worker with the session radix cache (sglang#27058).
# No Dynamo, no router, no NATS. Serves the native /generate + /close_session API
# that replay_sglang_native.py drives.
#
#   ./run_sglang_native.sh                  # GLM-4.7-Flash TP2 (matches the captured trace)
#   MODEL=Qwen/Qwen3-0.6B TP=1 GPU=0 ./run_sglang_native.sh   # tiny model, single GPU
set -euo pipefail
MODEL="${MODEL:-zai-org/GLM-4.7-Flash}"
TP="${TP:-2}"
GPU="${GPU:-0,1}"
PORT="${PORT:-30000}"
RADIX="${RADIX:-1}"   # 1 = --enable-session-radix-cache (the feature under test)
# Optional: shrink the KV pool so the trace saturates on a big-VRAM/small-model box.
MEM_FRACTION="${MEM_FRACTION:-}"

ARGS=()
[ "$RADIX" = 1 ] && ARGS+=(--enable-session-radix-cache)
[ -n "$MEM_FRACTION" ] && ARGS+=(--mem-fraction-static "$MEM_FRACTION")

echo "[run_sglang_native] model=$MODEL tp=$TP gpu=$GPU port=$PORT radix=$RADIX"
CUDA_VISIBLE_DEVICES="$GPU" exec python -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --tp "$TP" --page-size 16 \
  --skip-tokenizer-init --trust-remote-code \
  --enable-metrics \
  "${ARGS[@]}"
