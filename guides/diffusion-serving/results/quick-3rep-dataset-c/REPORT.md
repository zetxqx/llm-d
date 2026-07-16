# Cost-aware routing benchmark — report

Generated 2026-07-16 21:33 UTC. Workload: **dataset_c.json** (mixed-resolution t2i, see `workloads/`). Pool: 3×H100 running `Qwen/Qwen-Image`, batch=1. Run: `preset=quick workload=dataset_c.json replicas=3 started=2026-07-14T22:00:00+00:00`.

| arm | routing policy |
|---|---|
| baseline | baseline (k8s Service) |
| cost-aware | diffusion-cost-aware (llm-d EPP) |

## Calibration

| bucket | measured service time (s) |
|---|---|
| 512² | 1.72 |
| 768² | 2.06 |
| 1024² | 4.15 |
| 1536² | 13.83 |

Mixed mean service time S_mix = **4.71 s** → 3-pod capacity ≈ **0.636 req/s**.

## Headline: latency at the same offered load

Median over repetitions; both arms replay the identical request
sequence and arrival timeline, so rows are directly comparable.

| offered rate (req/s) | p99 baseline → cost-aware (s) | p99 reduction | mean baseline → cost-aware (s) | mean reduction |
|---|---|---|---|---|
| 0.2545 | 19.1 → 14.7 | **23%** | 5.3 → 4.4 | 18% |
| 0.4454 | 26.8 → 17.0 | **37%** | 8.9 → 5.3 | 40% |
| 0.6363 | 21.1 → 21.3 | **-1%** | 8.7 → 7.7 | 11% |

![p99 latency vs offered rate](figures/1_p99_latency.png)

![mean latency vs offered rate](figures/2_mean_latency.png)

## baseline (k8s Service)

| offered rate (req/s) | reps | p99 latency s (median) | p95 latency s | mean latency s | throughput req/s | failed | tainted |
|---|---|---|---|---|---|---|---|
| 0.2545 | 1 | 19.1 | 15.0 | 5.3 | 0.266 | 1 |  |
| 0.4454 | 1 | 26.8 | 24.1 | 8.9 | 0.425 |  |  |
| 0.6363 | 1 | 21.1 | 20.2 | 8.7 | 0.604 |  |  |

## diffusion-cost-aware (llm-d EPP)

| offered rate (req/s) | reps | p99 latency s (median) | p95 latency s | mean latency s | throughput req/s | failed | tainted |
|---|---|---|---|---|---|---|---|
| 0.2545 | 1 | 14.7 | 14.0 | 4.4 | 0.272 |  |  |
| 0.4454 | 1 | 17.0 | 14.2 | 5.3 | 0.460 |  |  |
| 0.6363 | 1 | 21.3 | 17.9 | 7.7 | 0.614 |  |  |


## Notes

- Median over repetitions; points marked tainted (pod churn mid-run, spot preemption) are excluded from medians when clean reps exist.
- The request mix and Poisson arrival timelines are identical across arms (seeded), so rows at the same rate are directly comparable.
- Raw data: `<arm>/rate<r>_rep<n>.json` (bench aggregates), `.csv` (per-pod queue-depth timeseries), `.meta` (pod set, taint flag).
