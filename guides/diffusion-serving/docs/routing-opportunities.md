# Routing opportunities for diffusion & omni model serving on llm-d

How the llm-d inference scheduler could better serve vLLM-Omni workloads —
diffusion, TTS, and multi-stage omni pipelines — from wiring up signals that
already exist to genuinely novel scheduling research. Grounded in the cloned
repos at this workspace's pinned versions and in the live deployment findings
from
[`../k8s/llm-d-optimized-baseline-omni/README.md`](../k8s/llm-d-optimized-baseline-omni/README.md)
(plus the `llm-d-tts` and `llm-d-edit` stacks). Findings incorporated through
2026-07-06; per-topic deep dives are indexed in
[reading-list.md](reading-list.md).

## The organizing principle (added 2026-07-06)

vLLM-Omni stages come in exactly three execution types
(`vllm_omni/config/stage_config.py:167-173`), and **the right routing policy
for a pool falls out of its type** — knowable statically per model, without
probing any pod (see [checking-model-stages.md](checking-model-stages.md)):

| `execution_type` | Engine underneath | Legible to today's routers? | Right routing signal |
| --- | --- | --- | --- |
| `LLM_AR` (Thinker, Talker, GLM-Image/BAGEL AR stage) | **real vLLM v1** (`OmniARScheduler` subclasses vLLM's scheduler) — emits `vllm:*` metrics, KV events, prefix caching | **yes** — everything llm-d/Dynamo/SMG already consume | cache/queue-affinity (Tier 3), queue depth |
| `LLM_GENERATION` (Code2Wav vocoder) | vLLM machinery, one-shot scheduler | partially (metrics exist; no KV story) | throughput/queue only |
| `DIFFUSION` (Z-Image, DiT stages) | **standalone `DiffusionEngine`** (`diffusion/diffusion_engine.py:101`) — no vLLM engine at all | **no** — `vllm:*` metrics structurally cannot exist | declared cost (Tier 2) |

One routing policy cannot cover an omni fleet; a hybrid request (GLM-Image,
BAGEL) crosses both regimes in a single request. That heterogeneity — not any
single missing metric — is the core thesis.

## Where we are today (verified 2026-07-02)

Deploying `Tongyi-MAI/Z-Image-Turbo` behind the llm-d router required falling
back to the chart's `payload-agnostic.yaml` profile, because:

1. **No parser** — the EPP has no request-body parser registered for
   `/v1/images/generations` (observed live: `HTTP 400: no parser registered
   matching path suffix`). The passthrough parser works but discards the body.
2. **No backend signal** — every stock scorer reads vLLM *LLM* metrics
   (`num_requests_waiting`, KV-cache utilization, prefix-cache state) that the
   vLLM-Omni diffusion path never emits.

Result: routing today is `active-request-scorer` (EPP-side in-flight count) +
`session-affinity-scorer`. Functional, but blind to everything diffusion-specific.
Every tier below removes one layer of that blindness.

**Scope refinement (2026-07-06):** the "no backend signal" blindness applies
to `DIFFUSION` stages *by construction* — the diffusion engine is not vLLM,
so the metrics can never appear there; they must come from vLLM-Omni's own
`vllm_omni:*` namespace (Tier 1a). `LLM_AR` stages (the TTS talker, omni
Thinkers) are ordinary vLLM underneath and are **already legible** to the
stock scorers — for those pools the problem is not signal absence but signal
*routing*: an LLM-metrics-only view of a colocated omni pod watches the wrong
stage (the Qwen3-Omni perf blog shows speech stages saturating while the
Thinker has headroom).

---

## Tier 1 — Signals exist but aren't connected (near-term, cheap)

### 1a. Export saturation gauges from vLLM-Omni

vLLM-Omni already maintains a dedicated Prometheus namespace — `vllm_omni:*` —
in [`vllm_omni/metrics/definitions.py`](../vllm-omni/vllm_omni/metrics/definitions.py)
("single source of truth" for pipeline, stage-gen-time, and cross-stage
transfer families; consumed by `vllm_omni.metrics.prometheus` for the
server-side `/metrics` pipeline). What's **missing** is the diffusion analogue
of vLLM's saturation gauges:

- in-flight diffusion jobs (≈ `vllm:num_requests_running`)
- queued diffusion jobs (≈ `vllm:num_requests_waiting`)
- estimated backlog in GPU-seconds (novel; see Tier 2)

If vLLM-Omni exported the first two, the scheduler's existing
`queue-scorer` / `load-aware-scorer` would light up with only a metric-name
mapping. **Contribution target: vllm-omni.**

### 1b. A diffusion request parser for the EPP

The scheduler registers parsers per path suffix —
`openai-parser`, `anthropic-parser`, `vertexai-parser`, `vllmhttp-parser`,
`vllmgrpc-parser`, `passthrough-parser` (see plugin registry strings in
`llm-d-inference-scheduler/pkg/epp/framework/plugins/`) — but none for
`/v1/images/generations` or `/v1/videos`. A small "openai-images parser" that
extracts `model`, `prompt`, `n`, `size`, and `num_inference_steps` would
restore body-aware routing that passthrough throws away today.
**Contribution target: llm-d-inference-scheduler.**

### 1c. Modality-aware steering (free body signal, zero engine changes)

The same `modalities` field the engine trusts for stage selection
(`entrypoints/openai/serving_chat.py:406-408`; backwards final-stage scan at
admission) sits in plaintext in the request body. An EPP scorer can know at
routing time whether a request stops at the Thinker or engages the speech
stages — and text-only traffic **is** affected by audio traffic on shared
replicas (shared Thinker batch/KV; shared GPU in single-GPU deploys), with
no intra-instance modality priority existing as a remedy
([vllm-omni-stage-pipeline.md](vllm-omni-stage-pipeline.md) §5). Steering
text-only requests to least-audio-loaded replicas is the cheapest possible
demo of body-aware omni routing: mixed text/audio workload, compare text-only
p99 with and without steering. **Contribution target:
llm-d-inference-scheduler (parser + scorer only).**

---

## Tier 2 — Diffusion's killer advantage: cost is declared up front

**The strongest research opportunity here.** For LLMs, output length is
unknowable at admission, so routing guesses. A diffusion request *declares its
cost in the body*:

```
cost ≈ num_inference_steps × f(resolution) × n
```

and per-step latency is stable/measurable — vLLM-Omni tracks the current step
as `denoise_step_idx` in
[`vllm_omni/diffusion/forward_context.py`](../vllm-omni/vllm_omni/diffusion/forward_context.py)
and its bench vocabulary already includes `denoise_step_latency_ms`
([`definitions.py`](../vllm-omni/vllm_omni/metrics/definitions.py)).

That enables two scorers no LLM can have:

- **Least-work-remaining routing.** Score endpoints by *estimated backlog
  GPU-seconds*, not request count. Two queued 20-step thumbnails ≠ one queued
  50-step 2048×2048 render. With diffusion latencies in the seconds-to-minutes
  range, one bad placement costs far more than for a ~200 ms LLM call — so the
  headroom over `active-request-scorer` should be measurable.
- **Predicted time-to-free.** Endpoints report `current_step / total_steps`
  per in-flight job → the router knows *when* each GPU frees up. The natural
  home is the scheduler's `latency-scorer`
  ([`pkg/epp/framework/plugins/scheduling/scorer/latency/plugin.go`](../llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/latency/plugin.go),
  "scores endpoints based on predicted latency headroom").

Dependencies: Tier 1a (backlog gauge) + Tier 1b (parser reads the declared cost).
Nothing in the scheduler does cost-declared scheduling today.

**Why the prediction is unusually sharp:** vLLM-Omni diffusion does no
cross-request batching — FIFO `RequestScheduler`, `max_num_running_reqs=1`
(`diffusion/diffusion_engine.py:141-144`) — so a worker's backlog is a strict
queue and `queue position × declared cost` is an (almost) *exact* wait-time
prediction, better information than any LLM router ever gets.

**Caveat (step caching, added 2026-07-06):** TeaCache and Cache-DiT — both
shipped in vLLM-Omni (`-O3` presets) — skip or cheapen denoise steps based on
*content-dependent* feature redundancy, turning the declared cost into an
**upper bound** with per-request variance (blog reports ~1.4–2.4× speedups).
Declared-cost routing stays correct (bounds still order endpoints), but a
refined estimator would learn each pool's average cache-speedup factor, and
the honest framing in any writeup is "declared upper-bound cost," not "exact
cost." Prompt-similarity step caching (NIRVANA-style) also feeds Tier 3.

---

## Tier 3 — Caches *do* exist for diffusion (just not KV caches)

The "diffusion has no cache to exploit" finding applies to token KV / prompt
prefix caches only. Real affinity signals exist, and the scheduler already has
scorers shaped for them:

| Scheduler plugin | Diffusion mapping |
| --- | --- |
| `mm-embeddings-cache-scorer` ([`scorer/mmcacheaffinity/scorer.go`](../llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/scorer/mmcacheaffinity/scorer.go) — "scores endpoints from multimodal encoder-cache match info") | Cached **text-encoder (T5/CLIP) prompt embeddings**. Image-gen usage is highly iterative — users tweak one word and regenerate — so prompt-embedding affinity should hit often. |
| `lora-affinity-scorer` | **Style-LoRA affinity**: route requests for a given LoRA to the endpoint that already has it loaded. For any multi-style image platform, likely the single highest-ROI scorer. |
| `session-affinity-scorer` (already active in our deployment) | img2img / iterative-editing chains stay on the endpoint that has the session's intermediates warm. |

Step-level DiT feature caches (TeaCache-style) keyed on prompt/seed/size are a
further affinity signal once servers expose hit predictions.

### 3b. AR-stage KV caches: one live today, one gated on RFC #1184 (added 2026-07-06)

For the `LLM_AR` half of the fleet, classic prefix-cache affinity is back on
the table — with a split status:

- **Live today: Qwen3-TTS.** `vllm_omni/deploy/qwen3_tts.yaml` ships the
  talker stage with **`enable_prefix_caching: true` in current main** (yaml
  comment: repeated text/voice/ref_audio across requests is a net win). The
  scheduler's stock prefix-cache scorers have real signal on our `llm-d-tts`
  stack *now* — no engine work needed.
- **Gated: Qwen3-Omni pipelines.** All three stages ship
  `enable_prefix_caching: false`, and the reason is structural, not a
  missing flag: the Talker's scheduler-visible "prompt" is **placeholder
  zeros** allocated purely for KV/batch-slot bookkeeping, while its real
  input is per-request Thinker hidden-state tensors riding
  `additional_information`
  ([vllm-omni-chunk-transfer.md](vllm-omni-chunk-transfer.md) §3b). Token
  prefix matching has nothing truthful to match on — that is the code-level
  root of [RFC #1184](https://github.com/vllm-project/vllm-omni/issues/1184)
  (hidden states returned only for the uncached suffix). Cache-affinity
  routing for omni AR stages is a *future* win gated on that RFC; for the
  Thinker stage specifically (normal token prompt) it becomes viable the
  moment the RFC's hidden-state fix lands.

---

## Tier 4 — Stage disaggregation: llm-d's P/D story maps onto vLLM-Omni's pipeline

vLLM-Omni's architecture *is* disaggregated stages — text-encode → DiT denoise
→ VAE decode, connected by OmniConnector, with a `PDDisaggregationMixin`
already in its entrypoints
([`vllm_omni/entrypoints/pd_utils.py`](../vllm-omni/vllm_omni/entrypoints/pd_utils.py)).
On the scheduler side, the multimodal disaggregation scaffolding exists too:
[`profilehandler/disagg/`](../llm-d-inference-scheduler/pkg/epp/framework/plugins/scheduling/profilehandler/disagg/)
with `always_disagg_mm_decider.go` and `multimodal_helpers.go` deciding
encode-vs-decode splits per request.

The diffusion analogue of prefill/decode disaggregation:

- **E/D/V split across heterogeneous pools** — text-encode on cheap GPUs, the
  DiT loop on big GPUs, VAE decode elsewhere — with the router picking a
  pair/triple per request and latents handed off like KV transfers.
- **Resolution-aware parallel-group routing** — requests routed to pools with
  the right sequence-/CFG-parallel group size for their resolution.
- **SLO-tiered routing** — turbo/low-step requests to a small-GPU pool,
  high-quality renders to the big pool; the "tier" is readable from the request
  body (Tier 1b parser again).
- **Async video jobs** — `/v1/videos` is job-based (submit → poll → fetch), so
  routing needs queue/drain awareness rather than request-response semantics; a
  fit for the scheduler's `flowcontrol` framework
  ([`pkg/epp/flowcontrol/`](../llm-d-inference-scheduler/pkg/epp/flowcontrol/)).

> **This is being built right now on the execution side.** vLLM-Omni RFC
> [#4590 "disaggregate diffusion inference"](https://github.com/vllm-project/vllm-omni/issues/4590)
> (open, high-priority, June 2026) proposes exactly the Encode / DiT /
> Decode(VAE) worker split — but its scheduling scope is *intra-instance*, and
> "scheduler awareness of stage readiness" is an unchecked open item. Nobody
> owns cluster-level routing across disaggregated diffusion stages. That seam —
> llm-d EPP routing over vLLM-Omni disaggregated stage pools — is where this
> investigation sits. See Related work below.

### In-tree evidence that stage pools are real, not hypothetical (added 2026-07-06)

- **Multi-stage hybrids ship today**: GLM-Image, BAGEL, and HunyuanImage-3.0
  are all AR→DiT pipelines in vLLM-Omni main. **BAGEL transfers KV cache
  between stages** (`omni_kv_config: need_send_cache / need_recv_cache`,
  criteria `prefill_finished` — `model_executor/models/bagel/pipeline.py`),
  i.e., a literal KV-transfer handoff between what would be separate pools —
  the exact shape of llm-d's P/D KV story. **HunyuanImage-3.0 ships
  standalone `hunyuan_image3_ar` / `hunyuan_image3_dit` configs** — the
  model is already packaged for split deployment.
- **Native disagg exists at the engine level**: `vllm serve --omni
  --omni-stage-id N --omni-master-address …` runs one stage per
  server/node (`DistStageRuntime` + Omni master discovery). The execution
  substrate for stage pools is shipped; only the cluster routing layer above
  it is missing.
- **The in-process precedent**: per-stage replicas (`stage_overrides:
  num_replicas`) form a `StagePool` with pluggable balancers — RANDOM /
  ROUND_ROBIN / `LEAST_QUEUE_LENGTH` (`stage_runtime.py:92-104`). The
  cluster-level equivalent of `LEAST_QUEUE_LENGTH` across disaggregated
  pools is exactly the unbuilt piece.
- **The transfer mechanics cooperate with routing**
  ([vllm-omni-chunk-transfer.md](vllm-omni-chunk-transfer.md)): inter-stage
  data moves as *publish-then-pull* — the producer `put()`s into a connector
  edge and gets back an opaque **address ticket**; any process holding the
  ticket and an edge connector can redeem it. Tickets are
  **replica-agnostic**, which is precisely the property a cluster router
  needs to choose *which replica* of stage N+1 pulls a given request's data
  at routing time. Request IDs are minted once at the front door and are the
  correlation spine end-to-end — any inserted routing tier must preserve
  them (Dynamo's does).
- **The seam to not fumble**: chunked streaming (`async_chunk`) is native
  vLLM-Omni's biggest latency win (first-audio 2790→655 ms) and is exactly
  what Dynamo's disagg mode sacrifices (one blocking hop per stage). A
  design that preserves park/poll/wake chunk flow across cluster-level stage
  pools would beat Dynamo on latency, not just on placement quality.

---

## Suggested experiments (in effort order)

Tooling note (2026-07-04): vLLM-Omni's own
`benchmarks/diffusion/diffusion_benchmark_serving.py` is reusable as the load
generator — Poisson `--request-rate`, `--max-concurrency`, heterogeneous
`--random-request-config` profiles, a real trace dataset, and built-in
`--slo`/`--slo-scale` with a linear resolution×steps cost model producing an
`slo_attainment_rate` metric. Run it in-cluster (it health-checks
`{base_url}/health`; don't run through a port-forward). Expected result
shape: load-aware routing mostly does **not** raise peak throughput — the
wins are p99, SLO attainment/goodput, and shifting the saturation knee.
Headline metric: **SLO attainment under heterogeneous load**.

1. **Baseline benchmark** — on the live `llm-d-omni` deployment, compare
   k8s-Service-random vs `random-picker` vs `active-request-scorer` under
   concurrent image-gen load (mixed sizes/steps). Establishes how much
   load-aware routing already buys.
2. **Modality-steering demo (Tier 1c)** — mixed text/audio workload against
   omni replicas; text-only p99 with vs without modality-aware steering.
   Cheapest body-aware demo; needs only a parser + scorer, no engine changes.
3. **In-flight gauge prototype** — patch vLLM-Omni to export
   running/queued-jobs gauges; map `queue-scorer` onto them; re-run (1).
4. **Work-remaining scorer prototype** — images parser (Tier 1b) + backlog
   GPU-seconds scoring (Tier 2); benchmark against (3) with a heterogeneous
   workload (mixed step counts and resolutions). This is the paper-worthy
   one; report declared cost as an upper bound if step caching is enabled.
5. **TTS prefix-affinity benchmark (Tier 3b)** — on `llm-d-tts`, repeated
   voice/ref_audio workload: stock prefix-cache scorer vs random across
   talker replicas. Zero engine work — the signal is already on in main.

## Related work (surveyed 2026-07-02)

> A fuller, annotated bibliography (including Cornserve, TetriServe, xDiT,
> NIRVANA, and the omni-serving systems surveyed 2026-07-03) lives in
> [reading-list.md](reading-list.md).

### Directly on E/D/V disaggregation

- **[vLLM-Omni RFC #4590 — "disaggregate diffusion inference"](https://github.com/vllm-project/vllm-omni/issues/4590)**
  (open, high-priority, filed June 2026). Splits Encode / DiT / Decode(VAE)
  onto separate workers. Reported numbers:
  - Steady-state throughput gain estimated **1.2–2.2×** — pipeline ceiling goes
    from `1/(T_enc + T_dit + T_dec)` to `1/max(...)`.
  - Overlapping DiT with VAE decode: single-request latency
    **9.84 s → 5.99 s (~1.64×)** (StreamWorld SP2).
  - Disaggregating the text encoder on a 24 GB 4090 (avoids offload): encoder
    latency **12.89 s → 0.40 s (~32×)**; DiT per-step 5.75 s → 3.76 s.
  - Proposed heterogeneous fleet: ~8× H100-class for DiT + 1–2 consumer GPUs
    (4090/5090) serving encode/decode for *many* DiT workers.
  - **Workload-dependent bottleneck flip:** DiT dominates image workloads
    (~76–87% of time, Qwen-Image @ 4 steps), but **VAE decode dominates some
    video workloads** — Wan2.2 TI2V-5B @ 121 frames: decode ≈ 51%, DiT ≈ 16%.
    This strengthens the Tier-4 case: the *right* stage split (and therefore
    routing) depends on the request, which only a body-aware router can see.
  - Open items include "scheduler awareness of stage readiness" — the
    integration point for llm-d.
- **[vLLM-Omni paper (arXiv:2602.02204)](https://arxiv.org/abs/2602.02204)** —
  "Fully Disaggregated Serving for Any-to-Any Multimodal Models." The stage-graph
  abstraction underlying the repo we deploy; intra-instance orchestration.
- **[DDiT (arXiv:2506.13497)](https://arxiv.org/abs/2506.13497)** — "Dynamic
  Resource Allocation for Diffusion Transformer Model Serving." Decouples DiT
  and VAE so each phase runs at its own parallelism degree; scheduler operates
  at **single-step granularity** (mid-generation GPU scaling). Up to **1.44×
  p99** latency improvement on text-to-video (T5 + OpenSora DiT/VAE).
- **[Katz (USENIX ATC '25)](https://www.usenix.org/system/files/atc25-li-suyi-katz.pdf)** —
  diffusion *workflow* serving; treats text encoder / denoiser / VAE as distinct
  components; latent parallelism for classifier-free guidance.

### Adjacent — supporting other tiers

- **[SwiftDiffusion (arXiv:2407.02031)](https://arxiv.org/abs/2407.02031)** —
  ControlNets-as-a-Service + LoRA-loading overlap, driven by production
  text-to-image traces; up to 7.8× latency reduction. Production evidence for
  **Tier 3's LoRA-affinity** value.
- **[DiffServe (arXiv:2411.15381)](https://arxiv.org/abs/2411.15381)**
  ([code](https://github.com/qizhengyang98/DiffServe)) — query-aware **model
  cascades**: cheap 2-step model first, escalate to 50-step model when a
  discriminator flags low quality. Cluster-level realization of **Tier 4's
  SLO-tiered routing**; up to 24% quality improvement at 19–70% lower latency
  violation rates.
- **[DistTrain (arXiv:2408.04275)](https://arxiv.org/abs/2408.04275)** — the
  *training*-side analogue: disaggregates modality encoder / LLM backbone /
  generator with independent parallelism per component.

### Positioning

The execution layer for diffusion stage disaggregation is actively being built
*inside* vLLM-Omni (RFC #4590), and single-instance schedulers (DDiT, Katz)
optimize within a replica. **The open gap is cluster-level, body-aware routing
across disaggregated diffusion stage pools** — llm-d's exact role in the
LLM P/D world, unclaimed in the diffusion world. Tiers 1–2 (declared-cost
routing) and Tier 4 (E/D/V pool routing) are the two halves of that claim.

> **Update 2026-07-03:** NVIDIA Dynamo (main) now ships a first-class
> vLLM-Omni backend including a **disaggregated multi-stage omni mode**
> (GLM-Image AR→DiT) — the execution plumbing for this seam. But its stage
> router is an explicit **round-robin message broker**
> (`dynamo/components/src/dynamo/vllm/omni/stage_router.py:96`), with KV-cache
> events disabled for omni workers and no cost/load/cache awareness. The
> unclaimed gap therefore narrows to its scheduling half: **intelligent stage
> routing** (declared-cost, load, cache affinity) — which is precisely the
> shape of llm-d's EPP scorer framework. Full analysis:
> [dynamo-diffusion-omni-report.md](dynamo-diffusion-omni-report.md).

> **Update 2026-07-05:** Surveyed SMG (lightseekorg Shepherd Model Gateway,
> SGLang-router lineage) as a third router data point. It has the most mature
> hybrid load model of the three — router-local in-flight counters + polled
> engine metrics (vLLM `/metrics`, 60 s) + an in-flight-since-poll delta that
> corrects scrape staleness, plus KV-event-driven cache routing over gRPC —
> but **zero omni support**: no `/v1/images/generations` / `/v1/audio/speech`
> / `/v1/videos` routes (404 fallback), and its entire cost model is
> token/prefix-denominated. Its `least_load` formula
> (`(queued + in-flight work)/throughput + kv_pressure`) is the Tier-2
> declared-cost scorer shape, in token units. Full analysis:
> [smg-router-analysis.md](smg-router-analysis.md).

> **Update 2026-07-06:** Deep-dived Dynamo's disagg omni internals
> ([dynamo-omni-stage-internals.md](dynamo-omni-stage-internals.md)); the
> competitive gap is wider than "round-robin router":
>
> - Dynamo **does not use vLLM-Omni's native disaggregation at all** — each
>   stage worker boots a full `AsyncOmni` tricked into single-stage mode via
>   a synthesized YAML, and Dynamo's own `OmniStageRouter` (a GPU-free
>   Python broker, **their invention** — vLLM-Omni ships no router-only
>   mode) re-implements the pipeline as sequential blocking hops.
> - The costs: no chunk streaming (stage N+1 waits for stage N to fully
>   finish), one request at a time per stage, and the final-stage output
>   returns as a raw `/dev/shm` handle that **pins the router to the
>   final-stage worker's node**.
> - **Deployment story: one bash script.** The only multi-stage example in
>   the repo is `disagg_omni_glm_image.sh` (single-node, experimental,
>   "may produce garbled outputs"); there is **no Kubernetes manifest for
>   multi-stage omni anywhere** — ~20 DGD manifests exist, all plain-LLM.
>   Our three llm-d stacks are already ahead of Dynamo on the cluster
>   deployment axis, before any routing intelligence is added.
> - Worth stealing: the **tickets-not-tensors** control-plane protocol
>   (connector address tickets accumulate through the router;
>   bulk data moves point-to-point).

> **Update 2026-07-06 (realtime/WebSocket):** Deployed the standalone router
> in front of a live Qwen3-Omni `/v1/realtime` pod (a since-removed
> experiment in the `omni` ns; the surviving stack is `realtime/`, which
> reproduced the same result 2026-07-14). **The EPP cannot route
> WebSocket upgrades today**: it schedules only at request-body
> `EndOfStream`, and treats header-only requests as complete only when the
> headers carry EoS — an upgrade GET keeps its stream open, so the EPP
> hangs and the handshake times out (Envoy's `upgrade_configs: websocket`
> is necessary but not sufficient). No upgrade handling exists anywhere in
> the EPP at HEAD. This adds a **Tier-1-class contribution target**:
> schedule `connection: upgrade` requests at header time — sound under
> payload-agnostic (parser and active-request-scorer never read the body),
> and the prerequisite for every connection-oriented routing idea in this
> doc's realtime column.

## Open questions

- ~~Does vLLM-Omni's text-encoder stage cache prompt embeddings across
  requests today?~~ **Answered (2026-07-04): no** — `encode_prompt()` runs
  per request inside the diffusion pipeline
  (`pipeline_z_image.py:241-324`); there is no cross-request embedding
  cache. Tier 3's embedding-affinity value therefore depends on *adding*
  one (or on RFC #4590's split making the encoder a shareable pool).
- What is the actual per-step latency variance across resolutions on L4 vs
  larger GPUs? (Determines how accurate declared-cost estimates can be.)
  With TeaCache/Cache-DiT enabled, also measure the cache-speedup variance —
  the gap between declared upper-bound and realized cost.
- For E/D/V disagg: is the latent handoff (DiT→VAE) small enough to cross
  nodes profitably, or is VAE colocated forever? (The connector layer
  already supports cross-node zero-copy RDMA — Mooncake transfer engine —
  so this is a bandwidth/latency question, not a plumbing one.)
- Can chunked streaming (`async_chunk`) survive a cluster-level stage
  router? The park/poll/wake receive path is per-engine and
  connector-agnostic, and tickets are replica-agnostic — nothing obviously
  forbids it — but nobody has built the control plane that forwards chunk
  tickets incrementally instead of per-stage-completion (Dynamo forwards
  once per stage).
- Where should the stage-graph knowledge live in llm-d? The topology is
  static per-model metadata (`pipeline.py` / deploy yaml) that a router can
  load at config time — per-pool `execution_type` → policy selection needs
  no discovery protocol.
