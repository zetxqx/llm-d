# Reading list: distributed diffusion & omni-model serving

Curated literature for the llm-d × vLLM-Omni routing investigation. Each entry
has a one-line "why it matters here." Survey dates: 2026-07-02 (first batch,
also cited in [routing-opportunities.md](routing-opportunities.md)) and
2026-07-03 (this consolidated list).

**Priority reads:** Cornserve (closest prior work to the thesis — differentiate
against it), TetriServe (overlaps the Tier-2 declared-cost idea), vLLM-Omni
paper + RFC #4590 (the execution layer we route over).

## 1. Omni / any-to-any model serving

| Paper | Venue/Date | What it does | Why it matters here |
| --- | --- | --- | --- |
| [vLLM-Omni](https://arxiv.org/abs/2602.02204) (arXiv:2602.02204) | Feb 2026 | Stage-graph disaggregation for any-to-any models; per-stage batching, inter-stage connectors; ≤91.4% JCT reduction | The system we deploy. Reviewer-noted gap: offline single workloads only — **no multi-tenant/bursty routing policies**. That gap is this investigation. |
| [Cornserve](https://arxiv.org/pdf/2512.14098) (arXiv:2512.14098) | Dec 2025 | ⭐ Cluster **planner** for generic any-to-any models: per-component placement/replication (e.g., 3× Thinker TP2 + 10× Talker+audio-gen on 16 GPUs); 2.68× throughput, 3–5.8× latency vs monoliths | **Closest prior work.** Solves offline *placement*; per-request *routing* across the placed replicas is the unclaimed half. Must cite & differentiate. |
| [EPD Disaggregation](https://arxiv.org/abs/2501.05460) | ICML 2025 | Encode–prefill–decode disaggregation for large multimodal models | Academic grounding for encoder disagg; Dynamo shipped this (NIXL embedding transfer). |
| [ElasticMM](https://openreview.net/pdf?id=Zd6VyjmN1S) | 2025 | Elastic multimodal parallelism; modality-aware load balancing across modality groups | Modality-aware *scaling*; complements modality-aware *routing*. |
| [HydraInfer](https://arxiv.org/abs/2505.12658) (arXiv:2505.12658) | May 2025 | Hybrid disaggregated scheduling for multimodal LLM serving | Another point in the disagg-scheduling design space. |
| VoxServe | 2026 | Streaming-centric serving for speech LMs | TTFA-centric serving — the metric our TTS streaming work targets. |

## 2. Diffusion serving systems & scheduling

| Paper | Venue/Date | What it does | Why it matters here |
| --- | --- | --- | --- |
| [DDiT](https://arxiv.org/abs/2506.13497) (arXiv:2506.13497) | Jun 2025 | Decouples DiT/VAE parallelism; **single-step-granularity** scheduler; ≤1.44× p99 on T2V | Step-granular scheduling = the intra-instance version of predicted time-to-free (Tier 2). |
| [TetriServe](https://arxiv.org/html/2510.01565v3) (arXiv:2510.01565) | Oct 2025 | ⭐ DiT serving for **heterogeneous image generation** (mixed resolutions/steps); integrates NIRVANA-style caching | Directly overlaps Tier-2 declared-cost routing — read before writing anything new. |
| [Katz](https://www.usenix.org/system/files/atc25-li-suyi-katz.pdf) | USENIX ATC '25 | Diffusion *workflow* serving; encoder/denoiser/VAE as components; latent parallelism for CFG | Component-level view of the pipeline this investigation routes across. |
| [SwiftDiffusion](https://arxiv.org/abs/2407.02031) (arXiv:2407.02031) | 2024 | ControlNets-as-a-Service + LoRA loading overlap, from production traces; ≤7.8× latency | Production evidence for Tier-3 LoRA-affinity routing. |
| [DiffServe](https://arxiv.org/abs/2411.15381) (arXiv:2411.15381, [code](https://github.com/qizhengyang98/DiffServe)) | 2024 | Query-aware **model cascades** (cheap model first, escalate on low quality) | Cluster-level SLO-tiered routing (Tier 4). |
| [vLLM-Omni RFC #4590](https://github.com/vllm-project/vllm-omni/issues/4590) | Jun 2026, open | E/DiT/VAE worker split; 1.2–2.2× throughput est.; VAE decode >50% of time for long video | The execution layer being built under Tier 4; "scheduler awareness of stage readiness" is an unchecked item = llm-d integration point. |

## 3. Intra-request diffusion parallelism (the layer *under* routing)

| Paper | Venue/Date | What it does | Why it matters here |
| --- | --- | --- | --- |
| [xDiT](https://arxiv.org/abs/2411.01738) (arXiv:2411.01738, [code](https://github.com/xdit-project/xDiT)) | 2024 | Hybrid SP + PipeFusion + CFG-parallel engine for DiTs; 13.29× at 4096px/16 GPUs | The parallelism engine a resolution-aware router would pick group sizes for. |
| [PipeFusion](https://arxiv.org/abs/2405.14430) (NeurIPS 2025) | 2025 | Patch-level pipeline parallelism reusing step-to-step stale activations | Lowest-communication DiT parallelism; why big renders scale across cheap interconnects. |
| DistriFusion | CVPR 2024 | Sequence parallelism w/ stale activations for U-Net diffusion | Baseline xDiT shows OOMs at high res — evidence the right parallelism **depends on the request** (Tier 4 routing input). |

## 4. Caching for diffusion serving

| Paper | Venue/Date | What it does | Why it matters here |
| --- | --- | --- | --- |
| [NIRVANA](https://arxiv.org/abs/2312.04429) (NSDI '24) | 2024 | Approximate caching: skip K denoise steps by reusing intermediate noise from **similar prompts**; 21% GPU savings in production; LCBFU eviction | Production proof that prompt-similarity affinity has real hit rates → Tier-3 embedding-affinity routing has signal. Follow-ups: TetriServe integration; Chorus (video). |

## 5. Adjacent / training-side

| Paper | Venue/Date | What it does | Why it matters here |
| --- | --- | --- | --- |
| [DistTrain](https://arxiv.org/abs/2408.04275) (arXiv:2408.04275) | 2024 | Disaggregated *training* for multimodal LLMs (encoder/LLM/generator) | The training-side analogue of stage disaggregation. |
| [Dynamo analysis](dynamo-diffusion-omni-report.md) (internal) | 2026-07-03 | NVIDIA Dynamo main: first-class vLLM-Omni backend, TTS, realtime audio, disagg stage router = **round-robin broker** | The competitive baseline to beat on stage routing. |
| [SMG analysis](smg-router-analysis.md) (internal) | 2026-07-05 | lightseekorg Shepherd Model Gateway: 3-tier cache routing (KV events → approx trees), hybrid ledger+polled-metrics load model, mesh tree-sync | Best-in-class *load* model, still token-denominated; **no omni endpoints at all** (404). Third proof the routing seam is unclaimed. |
| [vLLM-Omni stage pipeline](vllm-omni-stage-pipeline.md) (internal) | 2026-07-06 | Connectors (SHM/Mooncake-RDMA), stages-as-processes, `--omni-stage-id` native disagg, `modalities`→final-stage resolution, text/audio interference analysis | The execution substrate we route over; modality field = free routing signal; async_chunk is what naive disagg sacrifices. |
| [TTS vs diffusion](tts-vs-diffusion.md) (internal) | 2026-07-06 | AR (TTS) vs diffusion (T2I) architecture split, exceptions (GLM-Image hybrid, flow-matching TTS), serving-physics comparison table | Why one routing policy can't cover an omni fleet; the conceptual backbone of the three-stack matrix. |
| [Checking model stages](checking-model-stages.md) (internal) | 2026-07-06 | How-to: supported-models table → deploy yaml → pipeline.py `execution_type`; worked examples Z-Image (1-stage diffusion, no yaml) and Qwen3-TTS (AR→vocoder) | Stage topology is static per-model metadata a router can load; **TTS talker ships prefix caching ON in main** — live cache-affinity signal today. |
| [Dynamo omni stage internals](dynamo-omni-stage-internals.md) (internal) | 2026-07-06 | Code-level: each Dynamo stage worker boots a FULL `AsyncOmni` faked as single-stage; router passes connector *tickets* not tensors; internal request shapes; `--omni-router` = Dynamo-invented GPU-free broker (vLLM-Omni has none) | The closest existing analogue to an llm-d stage-routing tier — round-robin brain, SHM-pinned, no chunk streaming; the tickets-not-tensors protocol is worth stealing. |
| [vLLM-Omni chunk transfer](vllm-omni-chunk-transfer.md) (internal) | 2026-07-06 | Edges (YAML `input_connectors` → `(from,to)` connector map), `{req_id}_{stage_id}_{chunk_id}` keying, 5-act receive path (park `WAITING_FOR_CHUNK` → poll → fetch → use → wake), both edges walked: Talker→Code2Wav (tokens → `prompt_token_ids`) and Thinker→Talker (hidden states → `additional_information`, placeholder-zeros prompt) | The machinery streaming overlap is made of (what Dynamo's disagg lacks); the placeholder-prompt trick is the code-level root of RFC #1184's broken prefix caching. |

## The gap this list frames

Covered by prior work: intra-request parallelism (xDiT/PipeFusion), single-instance
stage scheduling (DDiT, Katz, TetriServe), cluster **placement** (Cornserve),
caching (NIRVANA), execution-layer disaggregation (vLLM-Omni, RFC #4590, Dynamo).

**Not covered anywhere:** per-request, body-aware **routing** across
disaggregated omni/diffusion stage pools — the live layer between Cornserve's
static planner and vLLM-Omni's executor. Dynamo's shipped answer is round-robin.
That is the contribution surface for llm-d (see
[routing-opportunities.md](routing-opportunities.md), Tiers 1–2 and 4).
