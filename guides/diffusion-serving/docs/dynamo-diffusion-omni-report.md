# Dynamo × vLLM-Omni: step-by-step — every routing & serving optimization on the omni path

**Analyzed:** `dynamo/` clone at commit `c914348c7` (main, 2026-07-03).
This walks an omni/diffusion request through Dynamo end to end and records, at
each step, what Dynamo optimizes (and what it doesn't). All claims carry file
references into the clone.

```
[0 build] → [1 deploy+register] → [2 HTTP in] → [3 pick worker] → [4 request plane]
         → [5 execute in worker] → [6 (disagg) stage hops] → [7 output] → [8 ops loop]
```

---

## Step 0 — Build & packaging

One image, many roles: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:<ver>` is built
from the official `vllm/vllm-openai` base (`container/context.yaml:56`), adds
Dynamo's Rust-runtime wheels + `components/src/dynamo/{common,frontend,vllm}`,
and pip-installs the **vLLM-Omni wheel** pinned at `v0.23.0rc1`
(`container/context.yaml:80`) via `container/deps/vllm/install_vllm_omni.sh`.

Optimizations/safeguards at this step:
- **Protected-packages constraints** — freezes the preinstalled vllm/torch
  stack so vllm-omni's deps can't move it (one image serves both plain-vLLM and
  omni workers).
- **Build-time hot-patch** — cherry-picks vllm-omni commit `17cf60a` onto
  site-packages (rc1's `OmniRequest` signature broke *all* vLLM workers, since
  vllm-omni monkeypatches `vllm.v1.request.Request` at import time).
- Bundles `nats-server`, `etcd`, UCX/NIXL into the runtime image
  (`container/templates/vllm_runtime.Dockerfile:71-91`).

## Step 1 — Deployment & registration

The k8s operator reconciles a `DynamoGraphDeployment`; each service is the same
image with a different `command` (`python -m dynamo.vllm.omni [--stage-id N |
--omni-router | --realtime]`).

At worker startup (`components/src/dynamo/vllm/omni/main.py`):
- **Model prefetch** — `fetch_model(config.model)` downloads weights before
  serving (no first-request download stall).
- **Registration into discovery** — `register_model(ModelInput.Text,
  model_type, ...)` with `model_type` derived from output modalities
  (Images/Videos/Audios) and `worker_type=WorkerType.Aggregated` — the
  multi-stage pipeline is one endpoint; no P/D split visible to the frontend.
  Discovery backends: `kubernetes | etcd | file | mem`
  (`components/src/dynamo/common/utils/runtime.py`).
- **DP-rank dedup** — non-leader data-parallel ranks skip endpoint
  registration entirely (`main.py`: "Non-leader DP rank; skipping"), so the
  router never sees duplicate endpoints per replica group.
- **Engine-aware health payload** — `VllmOmniHealthCheckPayload` probes the
  actual AsyncOmni engine, not just the process.

## Step 2 — Request enters the Rust frontend (the only HTTP hop)

The Rust frontend terminates OpenAI HTTP for all omni endpoints:
`/v1/images/generations`, `/v1/videos`, `/v1/audio/speech`
(`lib/llm/src/protocols/openai/audios.rs`), `/v1/chat/completions`.
Requests are validated and normalized into typed protocols
(`NvCreate{Image,Video,AudioSpeech}Request`) — i.e., unlike a passthrough
proxy, Dynamo's frontend **does parse omni request bodies**. (It just doesn't
*route* on them; see Step 3.)

## Step 3 — Worker selection (the routing decision)

`components/src/dynamo/frontend/main.py:216-236` — router modes:

| Mode | Signal | Works for omni workers? |
| --- | --- | --- |
| `round-robin` (**default**) | none | ✅ |
| `random` | none | ✅ |
| `power-of-two` | frontend-tracked load, 2 sampled candidates | ✅ |
| `least-loaded` | frontend-tracked load | ✅ |
| `device-aware-weighted` | device weights | ✅ |
| `kv` | KV-cache overlap events from workers | ❌ **inert for omni** |
| `direct` | caller-specified worker | ✅ (bypass) |

Key facts:
- **KV-aware routing — Dynamo's flagship — is disconnected on the omni path**:
  omni workers don't publish KV events
  (`base_handler.py:57` "TODO: Kv publishers not supported yet"; docs
  `docs/backends/vllm/vllm-omni.md` limitations: "KV cache events are not
  published for omni workers").
- Load-aware options (`least-loaded`, `power-of-two`) exist and work from
  frontend-side load tracking — but the **default is round-robin**, and the
  omni docs/launch scripts (`examples/backends/vllm/launch/agg_omni_*.sh`)
  don't set a mode.
- **Nothing reads the request body for routing**: no
  `num_inference_steps`/resolution/duration cost model anywhere in
  `lib/llm/src/kv_router/` or the frontend. A 4-step thumbnail and a 50-step
  4K render are identical to the router.
- Contrast: for VLM *understanding* requests, the frontend does deep
  body-aware routing — image `mm_hash` (xxh3), per-model image-token
  expansion, per-block MM metadata into the KV router
  (`lib/llm/src/preprocessor.rs`). None of that machinery is applied to
  generation/omni requests.

## Step 4 — Request plane (transport to the worker)

`docs/design-docs/request-plane.md`: requests travel over **TCP (default,
direct point-to-point)** or **NATS (brokered)**; workers are addressed as
`dyn://namespace.component.endpoint`. KV events (when used) ride a separate
event plane (**ZMQ or NATS**). Bulk tensors never touch these planes (Step 6).
Streamed responses (tokens, audio deltas) flow back over the same request-plane
connection.

## Step 5 — Execution in the worker

`OmniHandler` (`components/src/dynamo/vllm/omni/omni_handler.py`) translates
the typed request into vLLM-Omni inputs (`OmniTextPrompt`,
`OmniDiffusionSamplingParams`, per-stage `sampling_params_list`) and calls
**vLLM-Omni's own `AsyncOmni` orchestrator** (`base_handler.py:52`) — Dynamo
does not reimplement any inference. TTS specifics (voice/language validation
against the model's `config.json`, loaded at startup) live in
`audio_handler.py`.

Engine-level optimizations exposed as worker flags
(`docs/backends/vllm/vllm-omni.md:256-270`): `--enable-cpu-offload`,
`--enable-layerwise-offload` (DiT layerwise offload to shrink GPU memory) —
these forward to vLLM-Omni; the optimization itself is vLLM-Omni's.

## Step 6 — Disaggregated mode: stage-by-stage hops

With `--stage-id N` / `--omni-router`
(`stage_worker.py`, `stage_router.py`; reference: GLM-Image AR→DiT,
`examples/backends/vllm/launch/disagg_omni_glm_image.sh`):

1. Stage router receives the request, sends a `StageRequest` to a stage-0
   worker — chosen by **`client.round_robin(...)`** (`stage_router.py:96`).
2. Stage 0 runs its `AsyncOmni` (scoped to that stage), writes bulk output
   (hidden states/latents) into a **vLLM-Omni connector** (`connector.put()`),
   and returns only an opaque `stage_connector_refs` handle.
3. Router forwards the refs to a stage-1 worker (round-robin again); that
   worker fetches inputs from the connector, builds the engine-core request
   via `build_engine_core_request_from_tokens`, runs, and so on.
4. The final stage writes results to **shared memory** (`shm_write_bytes`);
   the router reads and formats them.

Real optimizations here:
- **Bulk data never transits the router** — only refs do ("pure message
  broker... never inspects or transforms inter-stage data",
  `docs/backends/vllm/vllm-omni.md:303`). Data plane = vLLM-Omni connectors +
  SHM; control plane = Dynamo request plane.
- **Independent per-stage scaling** — replicate the bottleneck stage (e.g., 3×
  DiT per 1× AR), impossible under single-process `vllm serve --omni`.
- **GPU isolation per stage** — each stage is its own process on its own GPU.

Not optimized: stage dispatch is round-robin — no load, cost, cache-affinity,
or stage-readiness signal. Known limitation: `async_chunk=true` (inter-stage
streaming) unsupported in disagg mode, so stage handoffs are bulk-synchronous.

## Step 7 — Output handling

`OutputFormatter` (`output_formatter.py`) + fsspec: generated images/videos are
**written to a filesystem/object store** (`--media-output-fs-url`: local, S3,
GCS, Azure) and returned as **URLs** (optionally rewritten to a CDN via
`--media-output-http-url`) instead of multi-MB base64 blobs riding the request
plane and frontend. `b64_json` remains available on request.

## Step 8 — Ops loop (after the request)

- **Metrics** — worker endpoint serves per-model labeled Prometheus metrics
  (`setup_metrics_collection`, `main.py`).
- **Health** — engine-probing health-check payload registered with the
  endpoint (restarts hit the engine, not just the HTTP port).
- **SLA planner / autoscaling** — Dynamo's planner scales
  DynamoGraphDeployment services on metrics; nothing omni-specific
  (`components/src/dynamo/planner/`), but omni workers participate as normal
  services.
- **Graceful shutdown** — signal handlers drain the endpoint before exit
  (`install_signal_handlers`, `graceful_shutdown=True` on `serve_endpoint`).
- **Realtime mode** — `realtime_handler.py` (added 2026-06-30): OpenAI
  Realtime WebSocket bridge onto `AsyncOmni.generate()` streaming; MVP,
  single-utterance.

---

## Scorecard: where the actual optimization lives

| Step | Optimization present | Routing intelligence |
| --- | --- | --- |
| 0 Build | constraint-pinned wheel, hot-patch, single multi-role image | — |
| 1 Register | weight prefetch, DP-rank dedup, engine health | — |
| 2 HTTP in | typed parsing of all omni bodies | parsed but **not used for routing** |
| 3 Pick worker | load-aware modes *available* | **default round-robin; KV router inert; no cost model** |
| 4 Transport | direct TCP default; separate event plane | — |
| 5 Execute | vLLM-Omni engine + offload flags (upstream's work) | — |
| 6 Stage hops | refs-not-data through router; per-stage scaling; SHM/connectors | **round-robin per stage** |
| 7 Output | media → object store + URL rewrite | — |
| 8 Ops | metrics, health, planner autoscaling, graceful drain | planner = capacity, not placement |

**Bottom line:** Dynamo's omni-path optimizations are concentrated in
*plumbing* (packaging, registration, transport, data movement, media handling,
ops) — all solid engineering. The two places where a *routing decision* is
made — frontend worker selection (Step 3) and disaggregated stage dispatch
(Step 6) — currently run **round-robin by default**, with the KV-aware router
explicitly disconnected for omni workers and no body/cost-aware scheduling
anywhere on the generation path.
