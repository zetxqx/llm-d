# Diffusion serving (text-to-image) with cost-aware routing

Serve text-to-image diffusion models (`Qwen/Qwen-Image` on [vLLM-Omni](https://github.com/vllm-project/vllm-omni)) behind the [llm-d inference scheduler](https://github.com/llm-d/llm-d-inference-scheduler), routing each request by its **declared cost**. The guide ships the deployment manifests for the model pool and the router, plus a self-contained A/B benchmark that measures what cost-aware routing buys over a plain Kubernetes Service under mixed-resolution t2i traffic.

## Why this benchmark exists

Diffusion requests are special: **their compute cost is declared up front.**
The request body carries `size` (width×height), `num_inference_steps`, and
`n` — enough to compute the request's cost before running it
(`cost ≈ steps × pixels × n`). LLM routers have to *estimate* remaining work;
a diffusion router can just read it. At batch=1 FIFO serving, the sum of
declared costs queued on a pod is an almost exact predictor of how long a new
request will wait there. (Step-caching features like TeaCache make this an
upper bound rather than an exact value, which is still useful for ranking
endpoints.)

The benchmark assumes **no batching** (each pod serves one request at a
time), which is vLLM-Omni's default serving mode.

The workload that exposes this is **Dataset C** from the vLLM-Omni
Qwen-Image performance dashboard
(`vllm-omni/benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md`)
— a weighted mixed-resolution text-to-image mix:

| bucket | steps | weight | declared cost (units¹) | share of total work² |
|---|---|---|---|---|
| 512×512 | 20 | 15% | 5.0 | ~3% |
| 768×768 | 20 | 25% | 11.25 | ~11% |
| 1024×1024 | 25 | 45% | 25.0 | ~42% |
| 1536×1536 | 35 | 15% | 78.75 | ~44% |

¹ 1 unit = one denoise step over one 1024×1024 megapixel (the EPP producer's
convention). ² by declared units; the measured-time split on A100 is similar.

The spread is the point: the largest requests are ~15× the smallest. A router
that is blind to cost treats a queued 512² thumbnail the same as a queued
1536² render, so short requests randomly get stuck behind long ones — exactly
the situation where tail latency collapses first.

## Choosing a workload

The routing gap grows with the *variance* of per-request service time
(CV² = squared coefficient of variation), so three workloads ship with
different jobs (CV² computed from measured H100 service times):

| file | mix | service-time CV² | use it for |
|---|---|---|---|
| `dataset_c.json` | 15/25/45/15 across 512²→1536² | 0.71 | the headline run — the upstream dashboard's official mixed workload |
| `dataset_bimodal.json` | 85% 512²×20st + 15% 1536²×50st | 2.12 | the strongest separation — thumbnails + whales, the worst case for cost-blind routing |
| `dataset_a.json` | 100% 512²×20st | 0 | negative control — with zero cost variance both arms should be identical; if they differ, something other than routing is leaking in |

Why the bimodal mix is the worst case: a 512² request takes ~1.7 s to serve,
but one 1536²×50st whale ahead of it in the queue costs ~20 s of waiting —
a single collision multiplies its latency by 10×. Random routing collides
constantly; cost-aware routing steers shorts away from whale-loaded pods. The mix keeps ~22 whales per 150-prompt run (stable
statistics) and roughly Dataset C's mean service time, so the same
calibration and rate grid apply — a difference in results between the two
workloads is a pure variance effect. Pass the workload as the second
argument: `scripts/run_all.sh full workloads/dataset_bimodal.json` — every
invocation gets its own run directory under `results/`, so runs with
different workloads (or repeats of the same one) never collide.

## The two arms

Both arms serve the same `Qwen/Qwen-Image` pool (3 replicas by default —
see `REPLICAS` in `scripts/env.sh` — 1×H100 each,
`vllm serve Qwen/Qwen-Image --omni`, **batch=1** — no `--step-execution`).

| arm | routing | what it shows |
|---|---|---|
| **A** baseline | one plain Kubernetes Service over the pool (no router). kube-proxy in its default iptables mode picks a uniformly *random* ready pod per new connection — not round-robin (that would need IPVS mode) | the out-of-the-box experience |
| **B** cost-aware | llm-d EPP: `openai-parser` + `diffusion-load-producer` + `diffusion-cost-scorer` (least outstanding declared cost) | the policy under test |

```
                       ┌───────────────────────────────┐
  bench Job ──────────►│ arm A: k8s Service            │──► pod 0 (H100)
  (Dataset C,          │   (random per connection)     │
   Poisson arrivals,   ├───────────────────────────────┤──► pod 1 (H100)
   in-cluster)         │ arm B: llm-d EPP              │
                       │   diffusion-cost-scorer       │──► pod 2 (H100)
                       └───────────────────────────────┘
```

Arm B's numbers include the EPP/ext-proc hop that arm A doesn't pay, so a
measured arm B win is net of the router's own overhead — if anything the
comparison understates B.

## Headline metrics

Load-aware routing usually does **not** raise peak throughput — the wins to
look for are:

- **p99 latency vs offered rate** (the money plot, log scale): tail latency
  is where cost-blind routing hurts — short requests stuck behind whales.
- **Mean latency vs offered rate**: the average user experience.
- **Per-pod backlog evenness** (`vllm_omni:num_requests_running+waiting`
  timeseries): arm A shows one pod piling up work while the other idles;
  arm B should keep the two lines close.

The load generator is vLLM-Omni's own
`benchmarks/diffusion/diffusion_benchmark_serving.py` (latency percentiles
and throughput come straight from its output JSON).

## Reproducibility

- The request **mix** is deterministic: the bench samples profiles with a
  fixed internal seed (42), so every arm and rate sees the identical request
  sequence for a given `--num-prompts`.
- The Poisson **arrival times** are drawn from Python's global RNG, which the
  upstream script does not seed. `scripts/run_bench.py` seeds it per
  repetition, so arrival timelines are also identical across arms.
- The pools run on spot H100s; `run_arm.sh` records the pod set before/after
  every measurement point and marks the point `tainted=true` in its `.meta`
  file if pods changed mid-run (spot preemption). Tainted points are excluded
  by the plots.

## Prerequisites

- The guide is self-contained in the llm-d repo (`guides/diffusion-serving`); the model-pool manifests are under `manifests/modelserver/`. Two external checkouts are passed in via env vars where needed: `VLLM_OMNI_DIR` (a [vllm-omni](https://github.com/vllm-project/vllm-omni) checkout, source of the benchmark client, needed by `calibrate.sh` and `run_arm.sh`) and `SCHED_DIR` (a `llm-d-inference-scheduler` checkout, needed by `build_epp_image.sh` only).
- A GKE cluster with the Gateway API Inference Extension CRDs, namespace `llm-d-omni`, and enough schedulable H100s in the spot pool for `REPLICAS` pods (`bobbm-spoth100` — edit the nodeSelector in `manifests/modelserver/qwen-image/base/` for your cluster).
- An EPP image built from `llm-d-inference-scheduler` branch `feat/diffusion-declared-cost`. **The older `images-gen-v2` tag does not contain the cost plugins** — arm B will crash-loop on it. Build with `scripts/build_epp_image.sh`.
- `kubectl`, `helm`, `envsubst`, `python3` locally; cluster egress to PyPI
  (the bench Job `pip install`s at startup).

## Quickstart

```bash
cd llm-d/guides/diffusion-serving

# 0. one-time: build + push the EPP image with the cost plugins
SCHED_DIR=/path/to/llm-d-inference-scheduler scripts/build_epp_image.sh

# 1. deploy the Qwen-Image pool (REPLICAS pods + --log-stats for the gauges)
scripts/deploy_pool.sh

# 2. sanity: /health must answer 200 through the arm B EPP (the bench gates
#    on it; arm A's plain Service hits the pods' /health directly)
scripts/switch_arm.sh b
kubectl run curl-check --rm -it --image=curlimages/curl -n llm-d-omni --restart=Never -- \
  curl -s -o /dev/null -w '%{http_code}\n' http://llm-d-omni-qwen-image-epp/health

# 3. everything else in one command: calibrate (first run only), sweep both
#    arms, render figures, write results/REPORT.md
export VLLM_OMNI_DIR=/path/to/vllm-omni   # source of the benchmark client
scripts/run_all.sh quick        # ~30-45 min end to end
scripts/run_all.sh full         # the real run (overnight)
```

Every `run_all.sh` invocation writes to its own run directory,
`results/<preset>-<workload>-<timestamp>/` (or pass an explicit label as the
third argument), with `results/latest` symlinking the newest run — so
repeated runs are always safe and old results are never clobbered. Re-running
with the *same* label resumes it: completed measurement points are kept, so
an interrupted sweep continues where it left off. Calibration is shared
across runs at the `results/` root (delete `results/capacity.env` to force
recalibration, e.g. after changing `REPLICAS`).

The pieces can also be run individually:

```bash
scripts/calibrate.sh                    # per-bucket service times -> capacity
RESULTS_DIR=results/myrun scripts/run_arm.sh a quick   # one arm, one preset
RESULTS_DIR=results/myrun scripts/run_arm.sh b quick   # gate: EPP logs show cost units
.venv/bin/python analysis/generate_report.py results/myrun   # figures + REPORT.md
```

The report lands in `results/<run>/REPORT.md`, figures in
`results/<run>/figures/`.

## Expected results (hypotheses, to be confirmed by the run)

- **Arm B vs A:** similar at very low load; from mid load on, arm B should
  show a visibly lower p99 and mean latency at every offered rate, with the
  gap widening as load approaches capacity. The queue-depth panels should
  show arm A occasionally stacking several large requests on one pod while
  the other drains.

## Interpreting failures

- Arm B EPP crash-loops → the deployed image lacks the cost plugins; rebuild
  (`scripts/build_epp_image.sh`) and check `scripts/verify_epp_image.sh`.
- `qwen-decode-shard-N has 0 endpoints` during calibration → pods were
  replaced (spot); `scripts/label_shards.sh` re-pins them.
- Bench exits immediately with a health error → `/health` is not passing
  through the router; check step 2 of the quickstart.

## Layout

```
workloads/        request mixes: dataset_c (headline), dataset_bimodal
                  (strongest separation), dataset_a (negative control)
manifests/
  baseline/       arm A: plain k8s Service over the pool (no router)
  router/         arm B: llm-d router values with the cost-aware plugins
  modelserver/    benchmark overlay: replicas + --log-stats (+ commented step-execution)
    qwen-image/   Qwen-Image t2i pool (base + gke), thin diff over guides/recipes
  calibration/    per-pod shard Services (calibrate.sh measures one pod directly)
  bench/          in-cluster bench Job template (bench + metrics-poller containers)
scripts/          build/verify EPP image, deploy, switch arms, calibrate, sweep
                  (run_all.sh = calibrate + both arms + report in one command)
analysis/         plot_results.py (figures), generate_report.py (REPORT.md)
results/          (gitignored) shared calibration at the root; one
                  subdirectory per run (raw JSON/CSV + report + figures),
                  results/latest -> newest run
```
