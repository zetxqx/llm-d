# Diffusion serving (text-to-image) with cost-aware routing

Serve text-to-image diffusion models (`Qwen/Qwen-Image` on [vLLM-Omni](https://github.com/vllm-project/vllm-omni)) behind the [llm-d inference scheduler](https://github.com/llm-d/llm-d-inference-scheduler), routing each request by its **diffusion cost**. A diffusion request carries `size`, `num_inference_steps`, and `n` in its body, so the router can compute its cost up front (`cost = steps x megapixels x n`) and send it to the endpoint with the least outstanding work. The guide ships the deployment manifests for the model pool and the router, plus a self-contained A/B benchmark against a plain Kubernetes Service.

## Workloads

| file | mix | source |
|---|---|---|
| `dataset_c.json` | mixed-resolution t2i, 15/25/45/15 across 512²→1536² | [Qwen-Image performance dashboard](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md) |
| `dataset_bimodal.json` | 85% 512²×20st + 15% 1536²×50st | derived for this guide — the worst case for cost-blind routing |

Pass the workload as the second argument to `run_all.sh`; every invocation gets its own run directory under `results/`.

## The two arms

Both arms serve the same `Qwen/Qwen-Image` pool (3 replicas, 1×H100 each, batch=1 — see `REPLICAS` in `scripts/env.sh`).

| arm | routing |
|---|---|
| **baseline** | plain Kubernetes Service over the pool (random pod per connection) |
| **diffusion-cost-aware** | llm-d EPP: `openai-parser` + `diffusion-load-producer` + `diffusion-cost-scorer` |

The cost-aware numbers include the EPP/ext-proc hop that the baseline doesn't pay, so a measured win is net of the router's own overhead.

## Prerequisites

- The guide is self-contained in the llm-d repo (`guides/diffusion-serving`); the model-pool manifests are under `manifests/modelserver/`. Two external checkouts are passed in via env vars where needed: `VLLM_OMNI_DIR` (a [vllm-omni](https://github.com/vllm-project/vllm-omni) checkout, source of the benchmark client, needed by `calibrate.sh` and `run_arm.sh`) and `SCHED_DIR` (a `llm-d-inference-scheduler` checkout, needed by `build_epp_image.sh` only).
- A GKE cluster with the Gateway API Inference Extension CRDs, namespace `llm-d-omni`, and enough schedulable H100s for `REPLICAS` pods (edit the nodeSelector in `manifests/modelserver/qwen-image/base/` for your cluster).
- An EPP image built from `llm-d-inference-scheduler` branch `feat/diffusion-declared-cost` (`scripts/build_epp_image.sh`); older images lack the cost plugins and the cost-aware arm will crash-loop.
- `kubectl`, `helm`, `envsubst`, `python3` locally; cluster egress to PyPI (the bench Job `pip install`s at startup).

## Quickstart

```bash
cd llm-d/guides/diffusion-serving

# 0. one-time: build + push the EPP image with the cost plugins
SCHED_DIR=/path/to/llm-d-inference-scheduler scripts/build_epp_image.sh

# 1. deploy the Qwen-Image pool
scripts/deploy_pool.sh

# 2. sanity: /health must answer 200 through the EPP
scripts/switch_arm.sh cost-aware
kubectl run curl-check --rm -it --image=curlimages/curl -n llm-d-omni --restart=Never -- \
  curl -s -o /dev/null -w '%{http_code}\n' http://llm-d-omni-qwen-image-epp/health

# 3. calibrate (first run only), sweep both arms, write results/<run>/REPORT.md
export VLLM_OMNI_DIR=/path/to/vllm-omni   # source of the benchmark client
scripts/run_all.sh quick        # ~30-45 min end to end
scripts/run_all.sh full         # the real run (overnight)
```

Re-running with the same run label resumes an interrupted sweep. Both arms replay the identical seeded request sequence and arrival timeline, so rows at the same rate are directly comparable; points hit by spot preemption are marked tainted and excluded.

## Results

Measured on 3×H100 (quick preset: 80 prompts per point, offered rates at 40/70/100% of the calibrated ~0.64 req/s capacity). Raw per-point JSON/CSV, full reports and figures are committed under [`results/quick-3rep-dataset-c/`](results/quick-3rep-dataset-c/REPORT.md) and [`results/bimodal-quick-1/`](results/bimodal-quick-1/REPORT.md).

Dataset C:

| offered rate (req/s) | p99 baseline → cost-aware (s) | p99 reduction | mean baseline → cost-aware (s) | mean reduction |
|---|---|---|---|---|
| 0.2545 | 19.1 → 14.7 | **23%** | 5.3 → 4.4 | 18% |
| 0.4454 | 26.8 → 17.0 | **37%** | 8.9 → 5.3 | 40% |
| 0.6363 | 21.1 → 21.3 | **-1%** | 8.7 → 7.7 | 11% |

Bimodal:

| offered rate (req/s) | p99 baseline → cost-aware (s) | p99 reduction | mean baseline → cost-aware (s) | mean reduction |
|---|---|---|---|---|
| 0.2545 | 35.6 → 19.8 | **44%** | 5.9 → 3.9 | 35% |
| 0.4454 | 30.0 → 20.4 | **32%** | 7.5 → 4.9 | 35% |
| 0.6363 | 41.3 → 21.7 | **47%** | 9.6 → 5.5 | 43% |

At 100% of capacity on Dataset C the p99s converge: at saturation every pod is always busy, so there is no routing freedom left — the mean is still 11% better. On the bimodal mix cost-aware routing also sustains 16% more goodput at the highest rate (0.601 vs 0.517 req/s).

## Layout

```
workloads/        request mixes (dataset_c, dataset_bimodal)
manifests/
  baseline/       baseline arm: plain k8s Service over the pool (no router)
  router/         cost-aware arm: llm-d router values with the cost plugins
  modelserver/    benchmark overlay: replicas + --log-stats
    qwen-image/   Qwen-Image t2i pool (base + gke), thin diff over guides/recipes
  calibration/    per-pod shard Services (calibrate.sh measures one pod directly)
  bench/          in-cluster bench Job template (bench + metrics-poller containers)
scripts/          build/verify EPP image, deploy, switch arms, calibrate, sweep
analysis/         plot_results.py (figures), generate_report.py (REPORT.md)
results/          calibration at the root; one subdirectory per committed run
```
