# How Dynamo runs vLLM-Omni stages: engines, inter-stage protocol, and the router

A code-level deep dive into Dynamo's disaggregated vLLM-Omni integration:
what each stage worker actually runs, how data moves between stages, what the
internal requests look like, and how many routers exist and who implemented
them. All file references into the `dynamo/` clone (HEAD `c914348c7`,
2026-07-03, current with origin/main as of 2026-07-06 — no omni-path commits
since the [dynamo-diffusion-omni-report.md](dynamo-diffusion-omni-report.md)
survey).

Companions: [vllm-omni-stage-pipeline.md](vllm-omni-stage-pipeline.md) (how
vLLM-Omni runs the same pipeline *natively*),
[routing-opportunities.md](routing-opportunities.md) (the routing layer this
informs).

The code lives in one module, `components/src/dynamo/vllm/omni/` (~4.4k
lines): `main.py` (mode dispatch), `stage_worker.py`, `stage_router.py`,
`types.py` (the wire protocol), `omni_handler.py` (aggregated mode),
`realtime_handler.py`.

## 1. One binary, four modes

`python -m dynamo.vllm.omni` dispatches on flags (`main.py:141-152`):

| Flags | Mode | What runs |
| --- | --- | --- |
| *(none)* | Aggregated omni worker | `OmniHandler` wrapping vLLM-Omni's full `AsyncOmni` — all stages in one process tree, like `vllm serve --omni` |
| `--stage-id N` | Single-stage worker | `OmniStageWorker` — ONE pipeline stage as an independent Dynamo endpoint |
| `--omni-router` | Stage router | `OmniStageRouter` — GPU-free pipeline broker |
| `--realtime` | Realtime bridge | bidirectional audio endpoint (Qwen3-Omni) |

The 2-stage GLM-Image deployment (`examples/backends/vllm/launch/
disagg_omni_glm_image.sh`) is therefore **4 processes**:

```
python -m dynamo.vllm.omni --stage-id 0      # AR worker,  CUDA_VISIBLE_DEVICES=0
python -m dynamo.vllm.omni --stage-id 1      # DiT worker, CUDA_VISIBLE_DEVICES=1
python -m dynamo.vllm.omni --omni-router     # OmniStageRouter — no GPU
python -m dynamo.frontend                    # Dynamo HTTP frontend (Rust)
```

## 2. Key finding: each stage worker boots a FULL vLLM-Omni, tricked into being single-stage

Dynamo does **not** use vLLM-Omni's native disaggregation
(`--omni-stage-id` / `DistStageRuntime` / Omni master server). Instead,
`_create_engine()` (`stage_worker.py:660-676`) takes the model's multi-stage
YAML, extracts only its own stage's config, rewrites it as
`stage_id: 0, final_output: True` with `runtime: {edges: []}`, writes a temp
single-stage YAML, and constructs a **complete `AsyncOmni` engine** on it:

```python
single_stage_config = {"stage_args": [stage_arg], "runtime": {"edges": []}}
...
return AsyncOmni(model=model, stage_configs_path=tmp_path)
```

So N stages = N independent `AsyncOmni` instances, each with its own
orchestrator thread and child stage process, each believing it serves a
one-stage model, each with `CUDA_VISIBLE_DEVICES` pre-narrowed by the launch
script (worker remaps device IDs to the visible set,
`_normalize_single_stage_runtime_devices`, `stage_worker.py:716-734`).

**vLLM-Omni's own orchestrator never sees the pipeline.** Dynamo dismembers
vLLM-Omni into per-stage engines and rebuilds the stage graph in its own
router — keeping only the OmniConnector layer from the original. A fair
amount of glue re-implements orchestrator behavior (processor-signature
dispatch across API versions `:282-338`, msgpack attribute-loss patches
`:578-632`), which helps explain the "experimental" label on GLM-Image.

## 3. How parameters pass between stages: two planes

### Control plane — small JSON through the router

The router walks stages **sequentially** (`stage_router.py:71-101`): call
stage 0, collect output, build next request, call stage 1, … choosing a
replica per stage via `client.round_robin(...)`. Everything through the
router is deliberately tensor-free: `request_id`, `original_prompt` (plain
dict), `sampling_params_list`, and the load-bearing field
**`stage_connector_refs`**.

### Data plane — tensors via vLLM-Omni's OmniConnector, bypassing the router

After a non-final stage finishes, the worker calls
`connector.put(from_stage, to_stage, request_id, payload)`
(`stage_worker.py:186-191`) with the stage's last `OmniRequestOutput`
(hidden states / token IDs / multimodal outputs). `put()` returns opaque
metadata — an **address ticket** — and *that ticket*, not the data, goes back
to the router inside `stage_connector_refs`. The next stage redeems it with
`connector.get(..., metadata=ref)` (`:363-364`), then runs the
model-specific processor (`ar2diffusion`, `thinker2talker`) or, with no
processor (the code2wav path), builds an `EngineCoreRequest` directly from
upstream token IDs (`:228-280`).

Ticket formats are connector-specific and opaque to the router
(`types.py:74-78`):

| Connector | Ticket |
| --- | --- |
| SHM | `{"shm": {"name": "<block>", "size": N}}` or `{"inline_bytes": …}` (small payloads) |
| Mooncake (RDMA) | `{"source_host": …, "source_port": …, "data_size": …}` |

The refs map **accumulates** as the pipeline progresses —
`{} → {0: ref0} → {0: ref0, 1: ref1}` — so any later stage can fetch from any
earlier stage, matching `engine_input_source` in the topology. If the stage
YAML omits connector edges, Dynamo synthesizes default `SharedMemoryConnector`
edges into a temp YAML (`_ensure_stage_connectors`, `stage_worker.py:494-565`).

## 4. The internal requests, concretely (GLM-Image, AR → DiT)

Wire types are pydantic models in `types.py`: `StageRequest` (worker input),
`StageOutput` (worker output; `extra="ignore"` drops unknown keys so stage
output can't accumulate arbitrary fields across stages).

**Router → stage 0** — the raw frontend request with a `request_id` stapled
on (`stage_router.py:85`, flagged in-code as a workaround):

```json
{ "request_id": "6f0e4c…", "model": "zai-org/GLM-Image",
  "prompt": "a corgi wearing sunglasses", "size": "1024x1024", "n": 1 }
```

**Stage 0 (AR) → router** — no tensors, just the ticket
(`stage_worker.py:205-215`):

```json
{ "original_prompt": { "prompt": "a corgi wearing sunglasses", "…": "tokenized fields, no tensors" },
  "stage_connector_refs": { "0": { "shm": { "name": "psm_a1b2c3", "size": 483210 }, "size": 483210 } },
  "sampling_params_list": { "0": { "temperature": 0.9 } },
  "finished": true }
```

**Router → stage 1 (DiT)** — built by `StageOutput.to_next_stage_request()`
(`types.py:84-94`), which forwards exactly three fields plus the id:

```json
{ "request_id": "6f0e4c…",
  "original_prompt": { "…": "…" },
  "stage_connector_refs": { "0": { "shm": { "name": "psm_a1b2c3", "size": 483210 } } },
  "sampling_params_list": { "…": "…" } }
```

**Final stage → router** — no downstream connector edge exists, so the worker
serializes the whole result to `/dev/shm` and returns the handle
(`stage_worker.py:218-226`):

```json
{ "shm_meta": { "name": "6f0e4c…", "size": 2914560 }, "finished": true }
```

The router `shm_deserialize()`s it and formats the OpenAI response
(`stage_router.py:142-170`). An in-code NOTE calls this a **single-node-only
workaround**: the router must share a machine with the final-stage worker; the
proper fix (a YAML-configured connector edge) is an open TODO.

## 5. Routers: how many, and who built the router-only mode

**Exactly one omni router per model pipeline** in every shipped example. But
there are two routing layers plus a mini-layer:

1. **Dynamo frontend** (Rust HTTP) — doesn't know pipelines exist. The
   `OmniStageRouter` registers itself as a plain `WorkerType.Aggregated`
   worker with `needs=[]` (`stage_router.py:205-217` — "the per-stage workers
   are private. From the frontend's topology view, the router serves
   end-to-end as Aggregated"). Stage workers explicitly do **not** register a
   model (`stage_worker.py:451-453`), so the frontend can't route around the
   router.
2. **OmniStageRouter** — the only component that knows the stage graph;
   walks it sequentially and formats the final response.
3. Within it, per-stage replica choice: `client.round_robin(stage_request)`
   (`stage_router.py:96`) across workers registered under the same
   `model_stage` name at `{namespace}.{model_stage}.generate`.

**Router-only mode is Dynamo's invention.** vLLM-Omni ships no standalone
router/orchestrator service: natively the pipeline brain is the orchestrator
*inside* the API-server process (single-command mode spawns stage child
processes under it; native disagg's `--omni-stage-id` workers register with an
"Omni master server," but the orchestrator+API server remain one coupled,
GPU-adjacent entrypoint). `OmniStageRouter` is ~240 lines of pure Python
broker — it imports vllm-omni only to parse the stage YAML
(`load_and_resolve_stage_configs`), loads no model, holds no GPU.

**Scalability of the router:** stateless across requests (the `stage_outputs`
list is per-call local), so N router replicas behind the frontend should work
in principle — but the SHM final-output workaround **pins the router to the
final-stage worker's node**, so the real current shape is one router,
colocated, single-node.

## 6. What this design costs vs. native vLLM-Omni

- **No chunk streaming.** The worker drains `engine.generate()` to a single
  `last_result` and does ONE `put()` per stage; the router does one
  request/response round trip per stage. Native vLLM-Omni's push-then-poll
  chunk protocol (`async_chunk`, `initial_codec_chunk_frames` — the
  TTFP 2790→655 ms win) has no equivalent: stage N+1 does not start until
  stage N fully finishes.
- **Single request at a time per stage** — stated in-code
  (`stage_worker.py:246-248`: "Dynamo processes one request at a time per
  stage"). Cross-request pipelining exists only insofar as different requests
  land on different replicas — chosen round-robin, blind to load/cost/cache.
- **Single-node final hop** (SHM handle to the router).
- **Re-implemented glue** tracking vLLM-Omni internals across versions — a
  maintenance surface vLLM-Omni's native orchestrator doesn't impose.

## 7. Routing implications (ties to the investigation)

- Dynamo's `OmniStageRouter` is the closest existing analogue to what an
  llm-d omni integration would build: a **GPU-free, pipeline-aware routing
  tier fronting private stage pools**. Dynamo has the tier — with a
  round-robin brain and a SHM tether.
- The llm-d opportunity is the same tier with real per-stage policies
  (declared-cost for DIFFUSION pools, cache/queue-affinity for LLM_AR pools —
  the `execution_type`-driven split in
  [checking-model-stages.md](checking-model-stages.md)) and a proper
  connector edge for final output so the router can live anywhere.
- The inter-stage protocol worth stealing: **tickets, not tensors, through
  the router.** The `stage_connector_refs` accumulation pattern keeps the
  routing tier body-light while the OmniConnector moves bulk data
  point-to-point — exactly the shape a cluster-level stage router wants.
- The two costs to avoid repeating: losing chunk streaming (the biggest
  native latency win) and coupling response formatting to node-local SHM.
