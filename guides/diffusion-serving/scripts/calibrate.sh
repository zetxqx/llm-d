#!/usr/bin/env bash
# Measure per-bucket service times on ONE pod (concurrency 1, direct to
# qwen-decode-shard-0, no router) and derive:
#   - S_mix  = sum(weight_b * S_b)      mixed mean service time
#   - mu     = REPLICAS / S_mix         pool capacity in req/s
# Writes results/capacity.env (sourced by run_arm.sh) and results/calibration.json.
# Per-bucket calib_*.json files are reused if present — copy them over from an
# old results dir to skip the ~10 min of re-measurement (per-pod service times
# don't change with replica count).
#
# Prereqs: deploy_pool.sh done. Applies the per-pod shard Services itself so
# it can target one pod directly — the EPP is NOT involved in calibration.
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${VLLM_OMNI_DIR:?set VLLM_OMNI_DIR to a vllm-omni checkout (source of the benchmark client)}"

mkdir -p "$BENCH_DIR/results"
kubectl apply -n "$NAMESPACE" -f "$BENCH_DIR/manifests/calibration/shard-services.yaml"
"$BENCH_DIR/scripts/label_shards.sh"

kubectl create configmap bench-scripts -n "$NAMESPACE" \
  --from-file="$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --from-file="$VLLM_OMNI_DIR/benchmarks/diffusion/backends.py" \
  --from-file="$BENCH_DIR/scripts/run_bench.py" \
  --from-file="$BENCH_DIR/scripts/collect_metrics.py" \
  --dry-run=client -o yaml | kubectl apply -f -

base_url="http://qwen-decode-shard-0.${NAMESPACE}.svc.cluster.local:8000"
declare -a widths=(512 768 1024 1536) steps=(20 20 25 35) weights=(0.15 0.25 0.45 0.15)
declare -a means=()

for i in "${!widths[@]}"; do
  w=${widths[$i]}; s=${steps[$i]}
  job="cab-calib-${w}"
  out="$BENCH_DIR/results/calib_${w}.json"
  if [[ ! -s "$out" ]]; then
    workload_b64=$(printf '[{"width":%s,"height":%s,"num_inference_steps":%s,"weight":1}]' "$w" "$w" "$s" | base64 -w0)
    echo ">>> calibrating ${w}x${w} @ ${s} steps (5 prompts, concurrency 1)"
    kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found >/dev/null
    JOB_NAME="$job" BASE_URL="$base_url" RATE="inf" NUM_PROMPTS="5" \
      ARRIVAL_SEED="1" POD_IPS="" MAX_CONCURRENCY=1 WORKLOAD_B64="$workload_b64" \
      envsubst '$JOB_NAME $BASE_URL $RATE $NUM_PROMPTS $ARRIVAL_SEED $POD_IPS $MAX_CONCURRENCY $WORKLOAD_B64' \
      < "$BENCH_DIR/manifests/bench/job.template.yaml" \
      | kubectl apply -n "$NAMESPACE" -f -
    kubectl wait --for=condition=complete "job/$job" -n "$NAMESPACE" --timeout=30m
    kubectl logs "job/$job" -n "$NAMESPACE" -c bench \
      | awk '/^=====RESULT_JSON=====$/{f=1;next} /^=====END=====$/{f=0} f' > "$out"
    kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found >/dev/null
  fi
  mean=$(python3 -c "import json; print(json.load(open('$out'))['latency_mean'])")
  means+=("$mean")
  echo ">>> ${w}x${w}: mean latency ${mean}s"
done

python3 - "$BENCH_DIR" "$REPLICAS" "${means[@]}" <<'EOF'
import json, sys

bench_dir = sys.argv[1]
replicas = int(sys.argv[2])
S = [float(x) for x in sys.argv[3:7]]
buckets = ["512", "768", "1024", "1536"]
weights = [0.15, 0.25, 0.45, 0.15]

S_mix = sum(w * s for w, s in zip(weights, S))
mu = replicas / S_mix
print(f"S_mix = {S_mix:.2f} s   mu({replicas} pods) = {mu:.3f} req/s")

with open(f"{bench_dir}/results/calibration.json", "w") as f:
    json.dump({"service_time_s": dict(zip(buckets, S)), "weights": dict(zip(buckets, weights)),
               "S_mix_s": S_mix, "replicas": replicas, "capacity_rps": mu}, f, indent=2)
with open(f"{bench_dir}/results/capacity.env", "w") as f:
    f.write(f"export CAPACITY_RPS={mu:.4f}\n")
print(f"wrote results/capacity.env (CAPACITY_RPS={mu:.4f}) and results/calibration.json")
EOF
