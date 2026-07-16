#!/usr/bin/env bash
# One-command driver: calibrate (if needed), sweep both arms, generate report.
#
#   ./run_all.sh <quick|full> [workload.json] [run-label]
#
# quick: ~30-45 min end to end. full: overnight.
# Assumes the pool is already deployed (scripts/deploy_pool.sh) — checked below.
#
# Every invocation gets its own run directory, results/<run-label>/ (label
# defaults to <preset>-<workload>-<timestamp>), so repeated runs never clobber
# or skip into each other's data. results/latest symlinks the newest run.
# Re-running with the SAME label resumes it: completed points are kept, so an
# interrupted sweep continues where it left off.
# Calibration artifacts (calib_*.json, calibration.json, capacity.env) live at
# the results/ root and are shared across runs — they depend on the hardware,
# not the run.
set -euo pipefail
source "$(dirname "$0")/env.sh"

preset="${1:?usage: run_all.sh <quick|full> [workload.json] [run-label]}"
workload="${2:-$BENCH_DIR/workloads/dataset_c.json}"
wl_tag=$(basename "$workload" .json | tr -c 'a-zA-Z0-9-' '-' | sed 's/-$//')
label="${3:-${preset}-${wl_tag}-$(date +%Y%m%d-%H%M)}"
export RESULTS_DIR="$BENCH_DIR/results/$label"
mkdir -p "$RESULTS_DIR"
ln -sfn "$label" "$BENCH_DIR/results/latest"
[[ -f "$RESULTS_DIR/run.info" ]] || \
  echo "preset=$preset workload=$(basename "$workload") replicas=$REPLICAS started=$(date -Is)" \
    > "$RESULTS_DIR/run.info"
echo ">>> run directory: results/$label"

# 0. Pool must be up: $REPLICAS ready replicas.
ready=$(kubectl get deploy optimized-baseline-omni-qwen-image-nvidia-gpu-vllm-decode \
  -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0)
if [[ "${ready:-0}" -lt "$REPLICAS" ]]; then
  echo "ERROR: model pool has ${ready:-0}/$REPLICAS ready replicas — run scripts/deploy_pool.sh first" >&2
  exit 1
fi

# 1. Calibrate once (measures per-bucket service times -> capacity/rate grid).
if [[ ! -f "$BENCH_DIR/results/capacity.env" ]]; then
  echo ">>> no results/capacity.env — running calibration first"
  "$BENCH_DIR/scripts/calibrate.sh"
fi
source "$BENCH_DIR/results/capacity.env"
echo ">>> capacity: $CAPACITY_RPS req/s (from results/capacity.env)"

# 2. Sweep both arms.
for arm in baseline cost-aware; do
  echo ""
  echo "============================================================"
  echo ">>> arm $arm ($preset)"
  echo "============================================================"
  "$BENCH_DIR/scripts/run_arm.sh" "$arm" "$preset" "$workload"
done

# 3. Figures + markdown report (self-bootstrapping venv: matplotlib is often
#    missing from system python, and corp pip mirrors may lack it — install
#    from PyPI directly; override with PIP_INDEX_URL if needed).
echo ""
echo ">>> generating report"
VENV="$BENCH_DIR/.venv"
if ! "$VENV/bin/python" -c "import matplotlib" 2>/dev/null; then
  echo ">>> bootstrapping $VENV (matplotlib, numpy)"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet \
    --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
    -r "$BENCH_DIR/analysis/requirements.txt"
fi
"$VENV/bin/python" "$BENCH_DIR/analysis/generate_report.py" "$RESULTS_DIR"

echo ""
echo ">>> done:"
echo "    report:  $RESULTS_DIR/REPORT.md"
echo "    figures: $RESULTS_DIR/figures/"
echo "    (also linked as results/latest)"
