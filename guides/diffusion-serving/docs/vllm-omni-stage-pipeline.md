# vLLM-Omni multi-stage pipelines: data flow, deployment modes, and modality routing

How vLLM-Omni runs a multi-stage omni model (Qwen3-Omni as the running
example: Thinker → Talker → Code2Wav), how stages exchange data, how to run
stages as separate services, and how a request's `modalities` field decides
which stages it touches. All file references into the `vllm-omni/` clone.
Companion docs: [diffusion-inference-walkthrough.md](diffusion-inference-walkthrough.md)
(what happens *inside* a diffusion stage) and
[routing-opportunities.md](routing-opportunities.md) (the routing layer above
all of this).

Context: the [Qwen3-Omni optimization blog](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization)
(2026-07-01) describes the perf work on this exact pipeline — per-stage CUDA
graphs (+299% throughput), async chunk (TTFP 2790→655 ms), async output,
stage replicas: 2.2 → 11.7 req/s overall at concurrency 64.

## 1. The pipeline shape

A multi-stage omni model is a list of stages, each a full vLLM-style engine
with its own scheduler, batching config, and GPU assignment, connected by
**connectors**. For Qwen3-Omni (`vllm_omni/deploy/qwen3_omni_moe.yaml`,
verified on 2× H100):

```
stage 0: Thinker  (LLM_AR, multimodal understanding + text gen)   devices: "0"
stage 1: Talker   (AR codec-token generator + code predictor)     devices: "1"
stage 2: Code2Wav (parallel vocoder, codes → waveform)            devices: "1"
```

Each stage gets independent `max_num_seqs`, `max_num_batched_tokens`,
`gpu_memory_utilization` (0.9 / 0.6 / 0.1 here — they share GPU 1),
`default_sampling_params`, and eager/cudagraph policy. Note
`enable_prefix_caching: false` on **all three stages** — prefix caching is
off for omni pipelines today (see RFC #1184: hidden-state I/O breaks it).

## 2. How stages pass data: the OmniConnector layer

Stages never call each other directly. The boundary is a key-value
**connector** (`vllm_omni/distributed/omni_connectors/connectors/base.py:12-76`):

```
put(from_stage, to_stage, put_key, data) -> (success, size, metadata)
get(from_stage, to_stage, get_key, metadata) -> (data, size) | None
```

keyed by `{request_id}_{stage_id}_{chunk_id}`. Implementations
(`factory.py:128-136`):

| Connector | Transport | Scope |
| --- | --- | --- |
| `SharedMemoryConnector` (default in deploy yamls) | POSIX `/dev/shm` + file locks; payloads ≤64 KB inlined (`shm_connector.py:17-100`) | same node |
| `MooncakeTransferEngineConnector` | RDMA, ZMQ handshake, zero-copy raw GPU tensors (`mooncake_transfer_engine_connector.py:70-98`) | cross-node |
| `MooncakeStoreConnector` | Mooncake distributed object store | cross-node |
| Yuanrong / Mori variants | NPU / XPU platform equivalents | cross-node |

**What crosses each boundary** is whatever the upstream stage's output spec
puts in the `OmniPayloadStruct` (`data_entry_keys.py:182-199`): for
Qwen3-Omni, hidden states + embeddings for Thinker→Talker, codec codes for
Talker→Code2Wav, plus `finished` / `is_segment_finished` flags. Which
connector feeds which stage is declared per stage in the deploy yaml:

```yaml
async_chunk: true
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra: {initial_codec_chunk_frames: 4, codec_chunk_frames: 25, ...}
stages:
  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_shared_memory
```

**Handoff protocol = push then poll** (`chunk_transfer_adapter.py:195-336`):
the producer stage's background save thread `put()`s each chunk as it is cut
(with `async_chunk: true`, the *first* audio chunk ships after just
`initial_codec_chunk_frames: 4` codec frames — this is the TTFP 2790→655 ms
optimization); the consumer stage's background load thread polls for its
next key, and the request sits in a `WAITING_FOR_CHUNK` state inside the
consumer's scheduler until the chunk lands. The connector store *is* the
inter-stage queue — there is no direct socket between engines.

`benchmarks/distributed/omni_connectors/cross_node_mooncake_transfer_engine.py`
is a standalone producer/consumer benchmark of the cross-node path (copy /
zero-copy / GPU-RDMA modes, throughput + MD5 verification).

For the layer below this — how edges are configured in YAML, the
`{req_id}_{stage_id}_{chunk_id}` key scheme, and the code-level
park/poll/wake receive path (`WAITING_FOR_CHUNK` →
`_poll_single_request` → prompt injection) — see
[vllm-omni-chunk-transfer.md](vllm-omni-chunk-transfer.md).

## 3. How stages are spun up (and how to run them separately)

**Key fact: stages are separate OS processes even in single-command mode.**
`vllm serve <model> --omni` starts an orchestrator thread
(`async_omni_engine.py:299-314`) which spawns **one child process per stage
replica** (`stage_runtime.py:562-573`), each with its GPU visibility scoped
to the yaml `devices:` field (`stage_runtime.py:307-322`; stages sharing a
GPU are initialized sequentially). Control plane between orchestrator and
stage processes is ZMQ; data plane is the connector.

**Native disaggregation — one stage per server/node:**

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --omni-stage-id 0 --omni-master-address <HOST> --omni-master-port <PORT>
# repeat with --omni-stage-id 1, 2 elsewhere; swap connector to Mooncake
```

`--omni-stage-id` flips the runtime from `StageRuntime` to `DistStageRuntime`
(`stage_runtime.py:1082-1096`; args parsed `async_omni_engine.py:246-272`);
each process registers with an **Omni master server** for discovery. This is
the machinery Dynamo's omni `--stage-id` worker mode wraps.

**Per-stage replicas** (`stage_overrides: num_replicas`, the blog's
1×Thinker + 2×Talker + 2×Code2Wav layout): replicas of a stage form a
`StagePool` with a pluggable balancer — `RANDOM`, `ROUND_ROBIN`, or
`LEAST_QUEUE_LENGTH` (`stage_runtime.py:92-104`). Slightly smarter than
Dynamo's stage router (round-robin only), still no cost/cache awareness.

## 4. How a request knows which stages to run (`modalities`)

Declarative match between what the request asks for and what each stage can
finalize:

1. **Client declares** `"modalities": ["text"]` (or `["text","audio"]`) in
   the chat-completions body.
2. **API server resolves** (`entrypoints/openai/serving_chat.py:406-408`):
   missing `modalities` defaults to *everything the model outputs* — for
   Qwen3-Omni that means audio ON unless you explicitly ask text-only.
3. **Stages declare terminal capability** in the model's pipeline definition
   (`model_executor/models/qwen3_omni/pipeline.py`): stage 0
   `final_output=True, final_output_type="text"` (:30-31); stage 1 not final;
   stage 2 `final_output_type="audio"` (:61-62). Same vocabulary across the
   zoo: `stage_configs/wan2_2_ti2v_dit_fp8.yaml:32` declares
   `final_output_type: video` on a DiT stage.
4. **At admission**, AsyncOmni walks the stage list **backwards** and stops
   at the first stage whose `final_output_type` is in the request's
   modalities (`async_omni.py:343-344` → `utils.py:590-632`):
   `["text"]` → final stage 0; anything with `audio` → final stage 2.
5. **The orchestrator enforces it**: outputs forward stage-to-stage only up
   to that request's final stage (`orchestrator.py:878-943`); the Thinker
   ships hidden-state chunks downstream only for requests continuing past
   stage 0.

Decided once at admission, per request, never per token — which is why mixed
modality requests can share the Thinker batch and diverge afterwards.

## 5. Does text-only traffic get affected by audio traffic? (Partial isolation)

**Yes, through two channels:**

1. **Shared Thinker.** All requests, regardless of modalities, run in the
   same stage-0 engine, continuous-batched together, sharing decode slots
   and KV memory. Audio requests load the Thinker exactly like text requests
   do; a text-only request queues behind them normally.
2. **Shared GPUs (deployment-dependent).** Talker/Code2Wav compute never
   involves text-only requests, but if those stages share a GPU with the
   Thinker (single-GPU deploys, e.g. `fish_qwen3_omni.yaml` all on
   `devices: "0"`), speech kernels steal SM time from Thinker decode and
   text-only latency degrades under audio load. The default 2-GPU layout
   (Thinker alone on GPU 0) removes this term.

**What text-only requests do NOT pay:** the speech stages themselves
(skipped per §4), and the audio requests' per-step hidden-state capture is
per-request work, mostly off the critical path since the async-output
optimization (inter-step GPU gap 2.8 ms → 41 µs per the blog).

**No intra-instance remedy exists**: there is no modality-aware priority,
admission, or batching inside vLLM-Omni. Protecting text traffic from audio
bursts has to happen a layer up — routing.

## 6. Routing implications (ties to the investigation)

- **Modality-aware routing is free signal:** the same `modalities` field the
  engine trusts for stage selection sits in plaintext in the request body.
  An EPP scorer can know at routing time whether a request stops at the
  Thinker or engages the speech stages — steering text-only traffic to
  text-only (or least-audio-loaded) replicas. Cheapest demo of body-aware
  routing: mixed text/audio workload, compare text-only p99 with and without
  modality steering.
- **Per-stage load is the real signal:** the blog shows speech stages
  saturate first while the Thinker has headroom — an LLM-metrics-only view
  of an omni pod watches the wrong stage (supports Tier 1: per-stage gauges).
- **The stage-replica balancer is another routing surface:**
  `LEAST_QUEUE_LENGTH` exists in-process; the cluster-level equivalent
  across disaggregated stage pools is unbuilt (Dynamo: round-robin).
- **Disagg currently sacrifices the biggest latency win:** Dynamo's
  disaggregated omni mode doesn't support `async_chunk` — and async chunk is
  worth 4× on first-audio (2790→655 ms). Preserving chunked streaming across
  cross-node stage pools is an open seam for any cluster-level design.
- **Prefix caching is off across all omni stages today**
  (`enable_prefix_caching: false` in the deploy yamls; RFC #1184 is the fix
  path) — so cache-affinity routing for omni AR stages is a *future* win,
  gated on that RFC.
