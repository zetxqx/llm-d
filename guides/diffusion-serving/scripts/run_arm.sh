#!/usr/bin/env bash
# Sweep driver: run one arm across a rate grid and harvest results.
#
#   ./run_arm.sh <a|b> <quick|full> [workload.json]
#
# quick:  rates {0.4, 0.7, 1.0} x capacity, 1 rep,  80 prompts  (~15 min/arm)
# full:   rates {0.3, 0.5, 0.7, 0.85, 1.0, 1.2} x capacity, 3 reps, 150 prompts
#
# Capacity (req/s) comes from results/capacity.env (written by calibrate.sh),
# else the CAPACITY_RPS default in env.sh. Results land in $RESULTS_DIR
# (exported by run_all.sh; defaults to results/adhoc for standalone use):
#   $RESULTS_DIR/<arm>/rate<r>_rep<n>.json   (bench aggregate metrics)
#   $RESULTS_DIR/<arm>/rate<r>_rep<n>.csv    (per-pod queue-depth timeseries)
#   $RESULTS_DIR/<arm>/rate<r>_rep<n>.meta   (pod UIDs; tainted if pods changed mid-point)
# Existing points are skipped, so re-running the same RESULTS_DIR resumes it.
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${VLLM_OMNI_DIR:?set VLLM_OMNI_DIR to a vllm-omni checkout (source of the benchmark client)}"

RESULTS_DIR="${RESULTS_DIR:-$BENCH_DIR/results/adhoc}"

arm="${1:?usage: run_arm.sh <a|b> <quick|full> [workload.json]}"
preset="${2:?usage: run_arm.sh <a|b> <quick|full> [workload.json]}"
workload="${3:-$BENCH_DIR/workloads/dataset_c.json}"

[[ -f "$BENCH_DIR/results/capacity.env" ]] && source "$BENCH_DIR/results/capacity.env"

case "$preset" in
  quick) fractions=(0.4 0.7 1.0);                 reps=(1);      num_prompts=80  ;;
  full)  fractions=(0.3 0.5 0.7 0.85 1.0 1.2);    reps=(1 2 3);  num_prompts=150 ;;
  *) echo "unknown preset: $preset" >&2; exit 1 ;;
esac

case "$arm" in
  a) base_url="$BASELINE_SVC_URL" ;;
  b) base_url="$EPP_SVC_URL" ;;
  *) echo "unknown arm: $arm" >&2; exit 1 ;;
esac

outdir="$RESULTS_DIR/$arm"
mkdir -p "$outdir"
workload_b64=$(base64 -w0 < "$workload")

"$BENCH_DIR/scripts/switch_arm.sh" "$arm"

# Publish the bench scripts as a ConfigMap (kept in sync with the local
# vllm-omni checkout on every run).
kubectl create configmap bench-scripts -n "$NAMESPACE" \
  --from-file="$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
  --from-file="$VLLM_OMNI_DIR/benchmarks/diffusion/backends.py" \
  --from-file="$BENCH_DIR/scripts/run_bench.py" \
  --from-file="$BENCH_DIR/scripts/collect_metrics.py" \
  --dry-run=client -o yaml | kubectl apply -f -

pod_state() {
  kubectl get pods -n "$NAMESPACE" -l "$POOL_SELECTOR" \
    --field-selector=status.phase=Running \
    -o jsonpath='{range .items[*]}{.metadata.uid}{" "}{.status.podIP}{"\n"}{end}' | sort
}

harvest() { # $1=jobname $2=container $3=start-marker $4=outfile
  kubectl logs "job/$1" -n "$NAMESPACE" -c "$2" \
    | awk "/^=====${3}=====$/{f=1;next} /^=====END=====$/{f=0} f" > "$4"
}

for frac in "${fractions[@]}"; do
  rate=$(python3 -c "print(f'{$frac * $CAPACITY_RPS:.4f}')")
  rate_tag=$(echo "$frac" | tr -d '.')
  for rep in "${reps[@]}"; do
    job="cab-${arm}-r${rate_tag}-n${rep}"
    out="$outdir/rate${frac}_rep${rep}"
    if [[ -s "$out.json" ]]; then
      echo ">>> $out.json exists, skipping"
      continue
    fi

    # Arm b: restart the EPP for a deterministic cold state per point (guards
    # against declared-cost counter drift from failed requests).
    if [[ "$arm" == "b" ]]; then
      kubectl rollout restart deploy/"$EPP_DEPLOY" -n "$NAMESPACE"
      kubectl rollout status deploy/"$EPP_DEPLOY" -n "$NAMESPACE" --timeout=5m
    fi

    before=$(pod_state)
    pod_ips=$(echo "$before" | awk '{print $2}' | paste -sd,)

    echo ">>> [$arm/$preset] rate=${rate} rep=${rep} prompts=${num_prompts} -> $job"
    kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found >/dev/null
    JOB_NAME="$job" BASE_URL="$base_url" RATE="$rate" NUM_PROMPTS="$num_prompts" \
      ARRIVAL_SEED="$rep" POD_IPS="$pod_ips" MAX_CONCURRENCY=64 WORKLOAD_B64="$workload_b64" \
      envsubst '$JOB_NAME $BASE_URL $RATE $NUM_PROMPTS $ARRIVAL_SEED $POD_IPS $MAX_CONCURRENCY $WORKLOAD_B64' \
      < "$BENCH_DIR/manifests/bench/job.template.yaml" \
      | kubectl apply -n "$NAMESPACE" -f -

    # Wait for the Job to finish either way; a point can legitimately take
    # ~num_prompts/rate seconds plus queue drain.
    if ! kubectl wait --for=condition=complete "job/$job" -n "$NAMESPACE" --timeout=2h 2>/dev/null; then
      echo "!!! $job did not complete; bench logs tail:" >&2
      kubectl logs "job/$job" -n "$NAMESPACE" -c bench --tail=20 >&2 || true
      kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found >/dev/null
      continue
    fi

    harvest "$job" bench RESULT_JSON "$out.json"
    harvest "$job" metrics-poller METRICS_CSV "$out.csv"

    after=$(pod_state)
    {
      echo "arm=$arm preset=$preset rate=$rate rep=$rep prompts=$num_prompts workload=$(basename "$workload")"
      echo "pods_before: $before"
      echo "pods_after:  $after"
      [[ "$before" == "$after" ]] && echo "tainted=false" || echo "tainted=true  # pod set changed mid-point (spot preemption?)"
    } > "$out.meta"
    [[ "$before" == "$after" ]] || echo "!!! TAINTED point (pod churn): $out" >&2

    kubectl delete job "$job" -n "$NAMESPACE" --ignore-not-found >/dev/null
    echo ">>> saved $out.{json,csv,meta}"
  done
done

echo ">>> arm $arm $preset sweep done -> $outdir"
