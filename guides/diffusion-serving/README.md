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

The gap between the two routing policies grows with how much per-request cost varies within the traffic, so two workloads ship with the guide:

| file | mix | use it for |
|---|---|---|
| `dataset_c.json` | 15/25/45/15 across 512²→1536² | the headline run — the upstream dashboard's official mixed workload |
| `dataset_bimodal.json` | 85% 512²×20st + 15% 1536²×50st | the strongest separation — thumbnails + whales, the worst case for cost-blind routing |

Why the bimodal mix is the worst case: a 512² request takes ~1.7 s to serve, but one 1536²×50st whale ahead of it in the queue costs ~20 s of waiting — a single collision multiplies its latency by 10×. Random routing collides constantly; cost-aware routing steers shorts away from whale-loaded pods. The mix keeps ~22 whales per 150-prompt run (stable statistics) and roughly Dataset C's mean service time, so the same calibration and rate grid apply. Pass the workload as the second argument: `scripts/run_all.sh full workloads/dataset_bimodal.json` — every invocation gets its own run directory under `results/`, so runs with different workloads (or repeats of the same one) never collide.

## The two arms

Both arms serve the same `Qwen/Qwen-Image` pool (3 replicas by default — see `REPLICAS` in `scripts/env.sh` — 1×H100 each, `vllm serve Qwen/Qwen-Image --omni`, **batch=1** — no `--step-execution`).

| arm | routing | what it shows |
|---|---|---|
| **baseline** | one plain Kubernetes Service over the pool (no router). kube-proxy in its default iptables mode picks a uniformly *random* ready pod per new connection — not round-robin (that would need IPVS mode) | the out-of-the-box experience |
| **diffusion-cost-aware** | llm-d EPP: `openai-parser` + `diffusion-load-producer` + `diffusion-cost-scorer` (least outstanding declared cost) | the policy under test |

```
                       ┌───────────────────────────────┐
  bench Job ──────────►│ baseline: k8s Service         │──► pod 0 (H100)
  (Dataset C,          │   (random per connection)     │
   Poisson arrivals,   ├───────────────────────────────┤──► pod 1 (H100)
   in-cluster)         │ cost-aware: llm-d EPP         │
                       │   diffusion-cost-scorer       │──► pod 2 (H100)
                       └───────────────────────────────┘
```

The cost-aware numbers include the EPP/ext-proc hop that the baseline doesn't pay, so a measured cost-aware win is net of the router's own overhead — if anything the comparison understates it.

## Headline metrics

Load-aware routing usually does **not** raise peak throughput — the wins to
look for are:

- **p99 latency vs offered rate** (the money plot): tail latency is where
  cost-blind routing hurts — short requests stuck behind whales.
- **Mean latency vs offered rate**: the average user experience.

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
- An EPP image built from `llm-d-inference-scheduler` branch `feat/diffusion-declared-cost`. **The older `images-gen-v2` tag does not contain the cost plugins** — the cost-aware arm will crash-loop on it. Build with `scripts/build_epp_image.sh`.
- `kubectl`, `helm`, `envsubst`, `python3` locally; cluster egress to PyPI
  (the bench Job `pip install`s at startup).

## Quickstart

```bash
cd llm-d/guides/diffusion-serving

# 0. one-time: build + push the EPP image with the cost plugins
SCHED_DIR=/path/to/llm-d-inference-scheduler scripts/build_epp_image.sh

# 1. deploy the Qwen-Image pool (REPLICAS pods + --log-stats for the gauges)
scripts/deploy_pool.sh

# 2. sanity: /health must answer 200 through the EPP (the bench gates on it;
#    the baseline's plain Service hits the pods' /health directly)
scripts/switch_arm.sh cost-aware
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
RESULTS_DIR=results/myrun scripts/run_arm.sh baseline quick     # one arm, one preset
RESULTS_DIR=results/myrun scripts/run_arm.sh cost-aware quick   # gate: EPP logs show cost units
.venv/bin/python analysis/generate_report.py results/myrun   # figures + REPORT.md
```

The report lands in `results/<run>/REPORT.md`, figures in
`results/<run>/figures/`.

## Results

Measured on 3×H100 (quick preset: 80 prompts per point, offered rates at 40/70/100% of the calibrated ~0.64 req/s capacity). Calibrated per-bucket service times: 512² = 1.72 s, 768² = 2.06 s, 1024² = 4.15 s, 1536² = 13.83 s. Raw per-point JSON/CSV, full reports and figures are committed under [`results/quick-3rep-dataset-c/`](results/quick-3rep-dataset-c/REPORT.md) and [`results/bimodal-quick-1/`](results/bimodal-quick-1/REPORT.md).

Dataset C:

| offered rate (req/s) | p99 baseline → cost-aware (s) | p99 reduction | mean baseline → cost-aware (s) | mean reduction |
|---|---|---|---|---|
| 0.2545 | 19.1 → 14.7 | **23%** | 5.3 → 4.4 | 18% |
| 0.4454 | 26.8 → 17.0 | **37%** | 8.9 → 5.3 | 40% |
| 0.6363 | 21.1 → 21.3 | **-1%** | 8.7 → 7.7 | 11% |

![p99 latency vs offered rate, dataset C](results/quick-3rep-dataset-c/figures/1_p99_latency.png)

![mean latency vs offered rate, dataset C](results/quick-3rep-dataset-c/figures/2_mean_latency.png)

Bimodal:

| offered rate (req/s) | p99 baseline → cost-aware (s) | p99 reduction | mean baseline → cost-aware (s) | mean reduction |
|---|---|---|---|---|
| 0.2545 | 35.6 → 19.8 | **44%** | 5.9 → 3.9 | 35% |
| 0.4454 | 30.0 → 20.4 | **32%** | 7.5 → 4.9 | 35% |
| 0.6363 | 41.3 → 21.7 | **47%** | 9.6 → 5.5 | 43% |

![p99 latency vs offered rate, bimodal](results/bimodal-quick-1/figures/1_p99_latency.png)

![mean latency vs offered rate, bimodal](results/bimodal-quick-1/figures/2_mean_latency.png)

At 100% of capacity on Dataset C the p99s converge: at saturation every pod is always busy, so there is no routing freedom left — the mean is still 11% better. On the bimodal mix cost-aware routing also sustains 16% more goodput at the highest rate (0.601 vs 0.517 req/s), because random routing wastes capacity when whole pods idle behind a whale pileup.

## Layout

```
workloads/        request mixes: dataset_c (headline), dataset_bimodal
                  (strongest separation)
manifests/
  baseline/       baseline arm: plain k8s Service over the pool (no router)
  router/         cost-aware arm: llm-d router values with the cost plugins
  modelserver/    benchmark overlay: replicas + --log-stats (+ commented step-execution)
    qwen-image/   Qwen-Image t2i pool (base + gke), thin diff over guides/recipes
  calibration/    per-pod shard Services (calibrate.sh measures one pod directly)
  bench/          in-cluster bench Job template (bench + metrics-poller containers)
scripts/          build/verify EPP image, deploy, switch arms, calibrate, sweep
                  (run_all.sh = calibrate + both arms + report in one command)
analysis/         plot_results.py (figures), generate_report.py (REPORT.md)
results/          shared calibration at the root; one subdirectory per run
                  (raw JSON/CSV + report + figures), all committed with the
                  guide; results/latest -> newest run (symlink, stays local)
```
