# Session-Control Benchmark: Setup and Workload

> Execution state, reproducibility manifest, and per-run records live in [`benchmark/LOG.md`](benchmark/LOG.md) — check there for where the campaign currently stands.

This document describes the benchmark setup for the llm-d session-control experiment ([llm-d-router#2003](https://github.com/llm-d/llm-d-router/issues/2003)): replaying a production agentic workload against a small SGLang fleet to measure how session-affinity routing affects prefix-cache hit rate, TTFT, and throughput under KV-cache pressure. The workload and methodology follow the [llm-d GLM-5.2 agentic serving blog](https://llm-d.ai/blog/serving-glm-5-2-agentic-workloads-on-llm-d), scaled down from a 64×H200 wide-EP deployment to an 8×H100 aggregated fleet.

## Benchmark Setup

### Hardware

| Item | Value |
|---|---|
| GPUs | 8× NVIDIA H100 80GB |
| Topology | 4 replicas × TP2 (2 GPUs per replica, 2 full nodes) |
| Platform | GKE, `nvidia-h100-80gb` spot pool (4 GPUs per node) |
| Serving mode | Aggregated (no prefill/decode disaggregation) |

Prefill/decode disaggregation is an explicit non-goal: session KV pinning is a decode-pod concern. Four replicas keep routing decisions observable (a wrong pick lands on one of three other pods) and, more importantly, place the fleet's KV-capacity knee at ~c64 — inside the concurrency sweep — rather than below it.

### Model

**[Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)** — 30.5B-parameter MoE (3.3B active, 128 experts / 8 routed), 48 layers, GQA with 4 KV heads, 262,144-token native context.

**Reason to choose this model.** The workload (below) imposes four hard requirements, and as of mid-2026 this model is the only small model that satisfies all of them simultaneously:

1. **≥256K native context.** The trace's median main-agent input is ~195K tokens (p95 >395K); the dataset ships 256K-filtered corpus builds that align exactly with this model's 262,144 window. This disqualifies Qwen3-32B (32K/131K), GLM-4.7-Flash (202K), and GLM-4.5-Air (128K).
2. **Standard full attention with token-indexed KV.** The experiment measures radix-tree prefix reuse and session-scoped eviction, and the trace models reuse as 64-token block hashes — both assume any shared prefix is reusable at arbitrary split points. The 2026 generation of small models uniformly broke this property to cut long-context KV cost: Qwen3-Coder-Next and Qwen3.6-35B-A3B use Gated DeltaNet linear attention (history compressed into fixed-size recurrent state), and Gemma 4 and Cohere North Mini Code use interleaved sliding-window attention (out-of-window KV discarded). On those architectures the cache-hit metric is not comparable to full-attention systems, and SGLang's session-radix path does not apply.
3. **Small-activation MoE.** A 195K-token prefill against a dense ~30B model costs ~10× the FLOPs of a 3.3B-active MoE; dense candidates (Qwen3.6-27B, Gemma 4 31B) cannot sustain the concurrency sweep.
4. **KV cost in the observable-scarcity range.** At 96KB/token (BF16) or 48KB/token (FP8), a median session holds 9–19GB of KV — expensive enough that the 8-GPU fleet enters cache pressure within the planned concurrency sweep, which is precisely the regime where affinity routing matters. Architectures with near-free KV (Gemma 4 at ~2KB/token) never reach scarcity at feasible concurrency, so routing gains are unmeasurable.

A secondary benefit: MORI (arXiv:2606.00866) benchmarks the same model family (Qwen3-30B-A3B, DP=3 TP=1 on H200) with self-collected Claude Code traces, giving us a published reference point for cross-checking system behavior.

### Benchmark Workload

**Weka trace** — the [SemiAnalysis `cc-traces-weka-*`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-061326) corpus (Apache 2.0): real Claude Code production sessions captured through a logging proxy and replayed offline. Traces store only per-request token counts and 64-token KV-block hashes — no prompt or completion text — so replay is model-agnostic: the harness synthesizes tokens to the recorded lengths and the block hashes reconstruct the exact prefix-reuse structure.

**General characteristics** (the `061326` build; other dated builds vary slightly):

| Property | Value |
|---|---|
| Traces (sessions) | 183 |
| Main-agent turns | 26,648 |
| Sub-agent groups | 652 (~3.6 per trace) |
| Total model requests | 44,990 |
| Total input tokens | ~9.3B |
| Total output tokens | ~42M |
| Median input length (main agent) | ~195K tokens |
| Median output length | ~317 tokens |
| Prefix reuse | 96% of turns reuse ≥90% of prior input as prefix |

Two structural properties make this workload the right stressor for session routing:

- **Long-prefix, short-output turns.** Each turn re-sends the accumulated session context plus a small delta; the value of a correct routing decision is the ~195K-token prefill that a cache hit avoids (~10s of TP2-H100 compute per miss).
- **Mid-stream branching.** Sub-agents fork from the parent session's intermediate context, so reuse is a tree, not a chain — this exercises radix-tree splitting and is where hybrid-attention caches (checkpoint-only reuse) diverge from full-attention behavior.

Requests exceeding the 262K window (long tail beyond the median) are handled by replaying the `-256k` filtered corpus builds, which the dataset publishes for exactly this purpose.

**Replay harness:** [kubernetes-sigs/inference-perf](https://github.com/kubernetes-sigs/inference-perf) `weka_trace_replay` (closed-loop per-session clients with recorded think times, hitting the router's OpenAI-compatible endpoint). Deployed in-cluster as a Job — the EPP is ClusterIP-only and each turn carries a ~1MB prompt body — via [`benchmark/inference-perf-weka.yaml`](benchmark/inference-perf-weka.yaml) (official `quay.io/inference-perf/inference-perf` image + ConfigMap). NVIDIA AIPerf's AgentX mode replays the same corpus and serves as a cross-check. Known limitation: inference-perf does not capture or echo response headers, so the treatment arm's `x-session-token` pinning does not engage from this harness yet — fine for the optimized-baseline arm, but the treatment arm needs a small token-echo patch to the harness before A-arm runs.

## llm-d Setup

### SGLang Setup

Image `lmsysorg/sglang:v0.5.15.post1`, 4 decode replicas, each launched as:

```
python3 -m sglang.launch_server
  --model-path Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
  --tensor-parallel-size 2
  --context-length 262144
  --kv-cache-dtype fp8_e4m3
  --mem-fraction-static 0.88
  --enable-session-radix-cache
  --radix-eviction-policy priority
  --enable-cache-report
  --enable-metrics
```

Key choices:

- **`--kv-cache-dtype fp8_e4m3`** halves per-token KV to 48KB. Per-replica KV pool ≈ 160GB × 0.88 − 30GB weights − ~6GB overhead ≈ **104GB ≈ 2.17M tokens**, i.e. ~11 fully-grown (195K) or ~17 time-averaged resident sessions per replica, ~44–68 fleet-wide. The fleet therefore transitions from all-resident (~c40) into eviction-active territory (~c64–c80), covering both regimes in one sweep.
- **`--enable-session-radix-cache` + `--radix-eviction-policy priority`** (sglang#27058/#29436 lineage): session-tagged radix entries with soft eviction protection, driven by a top-level `session_id` on `/generate` and released via `/close_session`. Note for future upgrades: sglang#29173 (2026-08-02) moves session tracking to `UnifiedRadixCache` (requires `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`, drops the priority-policy requirement) and changes eviction semantics — results across that boundary are not directly comparable.
- **`--mem-fraction-static 0.88`** (down from the recipe default 0.9): headroom for 262K chunked-prefill activation spikes.

Deployment manifests: [`modelserver/gpu/sglang/`](modelserver/gpu/sglang/) in this guide.

### llm-d Router Configuration

llm-d-router standalone v0.9.0 (EPP with Envoy sidecar). The experiment runs two router profiles against the same fleet, swapped via `helm upgrade` on the same release **followed by an explicit EPP restart** (`kubectl rollout restart deployment/session-control-epp`): the plugins ConfigMap is read at startup only — mounted-ConfigMap updates are not hot-reloaded, and helm does not roll the pod when only the ConfigMap changes. A missed restart silently runs the previous arm's config. The restart also clears EPP's in-memory state (approximate prefix index, in-flight load), which is desirable — every arm starts from identical cold router state, then goes through the same warmup pass.

**Treatment arm — session-control profile** ([`router/session-control.values.yaml`](router/session-control.values.yaml)):

| Plugin | Role | Weight |
|---|---|---|
| `session-id-producer` | Extracts `x-session-id` into the `SessionID` attribute (the seam for the #2003 binding tracker) | — |
| `session-affinity-filter` | Stateless pinning: first response returns `x-session-token` (base64 pod name); clients echo it and the filter narrows candidates to that pod | filter |
| `prefix-cache-scorer` | First-turn placement and fallback path | 3 |
| `queue-scorer` | Load balancing on first turn / fallback | 2 |
| `kv-cache-utilization-scorer` | Load balancing on first turn / fallback | 2 |

**Baseline arm — llm-d optimized-baseline profile** ([`router/optimized-baseline.values.yaml`](router/optimized-baseline.values.yaml), copied from `guides/optimized-baseline` and pointed at this fleet): llm-d's recommended default routing with no session awareness.

| Plugin | Role |
|---|---|
| `approx-prefix-cache-producer` | Router-side approximate index of each pod's cached prefixes |
| `inflight-load-producer` | Tracks in-flight token load per pod |
| `prefix-cache-affinity-filter` | Narrows candidates to pods with the longest approximate prefix match, subject to a prefill-throughput capacity model |
| `token-load-scorer` | Balances remaining candidates by outstanding token load |

This is a deliberately strong baseline: approximate prefix affinity should route most turns of a session to the pod holding its KV *when the router's index is accurate*. The experiment's hypothesis is that under KV pressure the router-side approximation goes stale (worker-side evictions are invisible to it — the failure mode ThunderAgent §3.2 documents for router-side radix trees), while explicit session pinning plus session-scoped eviction protection degrades more gracefully. The delta in cache-hit rate, TTFT, and sustainable concurrency between the two arms measures exactly that.

> **Calibration required:** the `prefix-cache-affinity-filter`'s `peakPrefillThroughput` default (15,928 tok/s) was calibrated for dense Qwen3-32B on H100 TP2. Qwen3-Coder-30B-A3B (3.3B active) prefills several times faster; run `guides/recipes/router/calibration` against this fleet and set the measured value before recording baseline numbers, or the filter will underestimate prefill capacity and skew placement.

## Planned Sweeps (TBD)

**Concurrency:** c16 → c32 → c48 → c64 → c72 → c80 (denser sampling near the predicted capacity knee at ~c64–c80; c128 requires either ~12 replicas or a HiCache tier and is out of scope for phase 1).

**Router arms** (same fleet, swapped via `helm upgrade` on the same release; run in this order):

| # | Arm | Values file | What it isolates |
|---|---|---|---|
| A | Treatment: session pinning | [`router/session-control.values.yaml`](router/session-control.values.yaml) | Explicit session affinity + scorers |
| B | Baseline: optimized-baseline | [`router/optimized-baseline.values.yaml`](router/optimized-baseline.values.yaml) | Approximate prefix affinity, session-unaware (calibrate `peakPrefillThroughput` first) |
| C | Ablation: scorer-only | [`router/scorer-only.values.yaml`](router/scorer-only.values.yaml) | Treatment profile minus pinning — plain scorer placement |

A−B measures what session pinning adds over llm-d's best session-unaware routing; A−C measures the value of pinning within the same plugin family; B−C measures the value of the router-side approximate prefix index by itself. Each arm change requires a fresh warmup pass (the fleet's radix caches must be repopulated) — never compare runs across a cold/warm boundary.

**Engine arms** (treatment router arm only): `--enable-session-radix-cache` on vs off, to separate router-level pinning gains from engine-level session-scoped eviction protection.

**Metrics:** prefix-cache hit rate (`cached_tokens` from `--enable-cache-report`), TTFT, ITL, step throughput, per-pod KV utilization and eviction counts, and per-pod request balance (to catch the affinity-vs-imbalance failure mode documented in ThunderAgent §3.2 / MORI §6.2.2).
