# Phase 3 details — disaggregated stage serving (code pointers)

Supporting detail for Phase 3 of [myplan.md](myplan.md). vLLM-Omni links are
pinned to commit
[`1b318d1`](https://github.com/vllm-project/vllm-omni/tree/1b318d11d17804c54c6ffa482efdd7abcb03657c)
(2026-06-28); llm-d-inference-scheduler paths were verified in the local
checkout (fork of `llm-d/llm-d-inference-scheduler`) — line numbers there may
drift on upstream main.

## 1. What vLLM-Omni's native disaggregation already does

vLLM-Omni can run one stage per node today. The pieces:

- **CLI**: worker nodes join with `vllm serve --omni --stage-id N
  --omni-master-address <head> --omni-master-port <p>`
  ([`entrypoints/cli/serve.py#L270`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/cli/serve.py#L270);
  headless worker entry `run_headless()` at L730). Replica ids are
  auto-assigned by the master (`--replica-id` is deprecated).
- **Head node** runs the API server, the orchestrator, and an
  `OmniMasterServer`
  ([`engine/stage_engine_startup.py#L141`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/engine/stage_engine_startup.py#L141))
  that workers register with over ZMQ; the head then drives every remote
  replica through pre-allocated input/output sockets
  (`DistStageRuntime`,
  [`engine/stage_runtime.py#L713`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/engine/stage_runtime.py#L713)).
- **Per-stage replica choice** is a `StagePool`
  ([`engine/stage_pool.py#L47`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/engine/stage_pool.py#L47))
  with pluggable balancers — random / round-robin / least-queue-length
  ([`distributed/omni_coordinator/load_balancer.py#L27`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/distributed/omni_coordinator/load_balancer.py#L27)).
  This is the in-process ancestor of what the EPP would do at cluster level.
- **Orchestrator**
  ([`engine/orchestrator.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/engine/orchestrator.py))
  owns the request lifecycle: admission, forwarding a finished stage's
  output to the next stage's pool (`_route_output` → `_forward_to_next_stage`
  → `StagePool.pick()`), and — important for us — with `async_chunk` on, it
  **pre-warms all downstream stages at admission** (`_prewarm_async_chunk_stages`,
  L1419) and then gets out of the way: chunks flow stage-to-stage directly
  through the connectors, not through the orchestrator.

So the missing piece is not stage execution — it's a **Kubernetes-native
version of the head node**: pools instead of StagePool, EPPs instead of the
built-in balancers, a service instead of the in-process orchestrator.

## 2. The data plane: OmniConnector

Inter-stage tensors (hidden states, codec chunks) move over **OmniConnector**
edges configured in the deploy yaml. Backends that exist today
([`distributed/omni_connectors/factory.py#L129`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/distributed/omni_connectors/factory.py#L129)):

| Backend | Transport | Scope |
| --- | --- | --- |
| `SharedMemoryConnector` | POSIX shared memory | same node (the default) |
| `MooncakeStoreConnector` | Mooncake distributed object store | cross-node |
| `MooncakeTransferEngineConnector` | Mooncake TE — RDMA or TCP, ZMQ control | cross-node |
| `MoriTransferEngineConnector` | AMD MORI | AMD GPUs |
| `Yuanrong*Connector` | Huawei YuanRong | Ascend NPUs |

**There is no NIXL backend in OmniConnector** — confirmed by search at the
pinned commit; adding one is a real contribution. (Separately, the AR
prefill/decode split *inside* an omni pipeline uses vLLM's native
`kv_transfer_config` connectors, where NIXL already exists — see
`PDDisaggregationMixin`,
[`entrypoints/pd_utils.py#L21`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/pd_utils.py#L21).
The OmniConnector NIXL work is for the omni-specific plane: hidden states and
chunk streams.)

### How data is handed off (and what a router must respect)

The pattern is **publish, then pull**: the producer `put()`s a payload and
gets back a small metadata dict (e.g. `{source_host, source_port}` for
Mooncake TE); the consumer later `get()`s using the key — and, for
point-to-point transports, that metadata
([`omni_connectors/connectors/base.py#L21`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/distributed/omni_connectors/connectors/base.py#L21)).
Payload keys are `{request_id}_{producer_stage}_{chunk_id}`
([`chunk_transfer_adapter.py#L304`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L304)).

Routing consequences, stated carefully:

- Keys contain no replica identity, so **any replica of the consumer stage
  can redeem a payload** — the router is free to pick the consumer pod per
  request.
- But **within one request the choice must be sticky**: chunked streams poll
  keys by a per-request counter, and the code warns that multi-replica
  deployments need sticky per-stream routing
  ([`chunk_transfer_adapter.py#L50-L57`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py#L50)).
- The producer's address is resolved per request (the orchestrator looks up
  which replica is bound to the request,
  [`orchestrator.py#L1505`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/engine/orchestrator.py#L1505))
  — a cluster coordinator must carry this producer info to the consumer,
  which is exactly the "pass the handle, not the tensor" rule.
- Request IDs are minted once at the front door and are the correlation key
  for every transfer — any routing tier must preserve them end-to-end.

### Cross-stage KV cache (hybrid models)

BAGEL's thinker→DiT edge really transfers KV cache
(`omni_kv_config: need_send_cache / need_recv_cache`,
[`model_executor/models/bagel/pipeline.py#L45`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/model_executor/models/bagel/pipeline.py#L45)).
Note: GLM-Image does **not** — its pipeline sets `need_recv_cache: False`
and uses an input processor instead
([`glm_image/pipeline.py#L44`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/model_executor/models/glm_image/pipeline.py#L44)).

## 3. RFC #4590 — Encode / DiT / Decode split for diffusion

[RFC #4590](https://github.com/vllm-project/vllm-omni/issues/4590) (open,
high priority, June 2026) splits the colocated diffusion pipeline into
independent Encode / DiT / VAE-Decode workers, targeting heterogeneous
fleets (DiT on H100-class, encode/decode on consumer GPUs). Status as of
2026-07-07:

- Linked [PR #3208](https://github.com/vllm-project/vllm-omni/pull/3208)
  (disaggregated encoder) reports ~10–15% throughput gain on short/mid-step
  workloads.
- Reported motivation numbers: ~1.2–2.2× steady-state throughput from
  pipelining; encoder off a 24 GB card 12.89 s → 0.40 s; DiT+VAE overlap
  9.84 s → 5.99 s. Bottlenecks flip by workload: DiT dominates images
  (~76–87%), VAE decode dominates some video (Wan2.2 TI2V @121 frames:
  decode ≈ 51%).
- **Every checklist item is still unchecked**, including "add scheduler
  state for stage readiness" — the natural integration point for a cluster
  router. Its scheduling scope is intra-instance; nobody owns cluster-level
  routing across the split. That seam is this plan's Phase 3.

## 4. What llm-d already has for disaggregation

Verified in the local checkout (paths relative to
`llm-d-inference-scheduler/`):

- **Disagg profile handler** — the EPP already plans up to three stages
  (Encode / Prefill / Decode) for one request in a single pass:
  `pkg/epp/framework/plugins/scheduling/profilehandler/disagg/disagg_profile_handler.go`
  (stage loop in `Pick()`; per-request "should we disaggregate?" decider
  plugins). Design doc with sequence diagrams: `docs/disaggregation.md`.
- **How picks travel**: the request is physically routed only to the
  terminal (decode) pod; the other picks ride as headers —
  `x-prefiller-host-port`, `x-encoder-hosts-ports`
  (`pkg/common/routing/common.go`).
- **Who executes the hops**: a **sidecar on the terminal pod**
  (`pkg/sidecar/proxy/`) reads the headers and runs the pipeline —
  encode fan-out, then prefill, then local decode — speaking pluggable KV
  connectors, with **NIXL as the default** (`connector_nixlv2.go`; also
  sglang, mooncake, shared-storage variants).
- **One pool per EPP**: an EPP watches exactly one InferencePool; stages are
  distinguished by `llm-d.ai/role` labels inside that pool. Separate pools
  per stage ⇒ one EPP per pool, plus something above them.
- **The coordinator seam already exists**: `Prefer: if-available` +
  HTTP 412 (`pkg/epp/requestcontrol/director.go`, release notes PR #1288) —
  an external coordinator can ask the EPP for a conditional decode pick and
  restart the pipeline when the EPP says the cache isn't there. Direct
  evidence the project anticipates an external cross-stage orchestrator.
- **Flow control** (`pkg/epp/flowcontrol/`): fairness / ordering (including
  SLO-deadline) / eviction plugins exist, but the entry point is blocking
  per-request (`EnqueueAndWait`) — an async job-handle entry point for
  `/v1/videos`-style work would be new.

## 5. Two architecture options for omni stage routing

**Option A — extend llm-d's existing pattern (one pool, role labels,
sidecar).** Stages live in one InferencePool with role labels; one EPP plans
all stage picks per request (a new omni profile handler generalizing the
E/P/D one); an omni-aware sidecar on the terminal stage executes hops,
carrying OmniConnector metadata instead of vLLM KV params.

- Pros: no new service; reuses shipped handler/sidecar/header machinery.
- Cons: the sidecar chain is synchronous per hop — as-is it would reproduce
  Dynamo's one-blocking-hop-per-stage behavior and lose chunked streaming;
  all stage picks are made once at admission (no mid-pipeline adaptation);
  and the sidecar protocol is KV/LLM-shaped today.

**Option B — coordination service + one EPP per stage pool** (the option in
the llm-d disagg design doc; recommended there — *unverified, the Google Doc
needs corp auth*). Each stage is its own InferencePool with its own EPP
(each pool gets the right policy for its stage type — Phase 1's cost scorer
for DiT pools, prefix-cache scorers for AR pools). A standalone coordination
service replays vLLM-Omni's orchestrator logic: admits the request, asks
each stage pool's EPP for a pod, passes connector metadata forward, and uses
the existing 412 contract for conditional picks.

- Pros: per-stage routing policies; matches vLLM-Omni's own orchestrator
  shape (including async-chunk pre-warming, which needs *all* stage picks up
  front — something Option A's handler can also do, but the coordinator can
  re-pick on failure); pools scale independently.
- Cons: a new service to build and operate; the EPP-per-pool fan-out adds a
  scheduling RTT per stage (mitigated by picking all stages at admission,
  which async-chunk pre-warm requires anyway).

Constraints either option must respect (learned from Dynamo's attempt, which
routes stages round-robin and drops streaming):

1. one request ID front door → final stage (it keys every transfer);
2. connector metadata, never tensors, through the control plane;
3. sticky consumer replica per request for chunked streams;
4. keep `async_chunk` alive — pre-warm downstream stages at admission and
   let chunks flow connector-to-connector; first-audio latency
   (2790 → 655 ms in native vLLM-Omni) is the number to protect.

## 6. Contribution list for Phase 3

| # | Contribution | Where | Status today |
| --- | --- | --- | --- |
| 1 | Per-stage k8s packaging (one pool per stage, head/worker manifests) | new (k8s/ here, then llm-d examples) | nothing upstream; Dynamo has no multi-stage k8s manifests either |
| 2 | NIXL backend for OmniConnector | vllm-omni `distributed/omni_connectors/` | does not exist (verified) |
| 3 | Omni stage profile handler or coordination service (Option A/B) | llm-d-inference-scheduler / new service | E/P/D handler + 412 seam exist to build on |
| 4 | Async job admission for `/v1/videos` | llm-d flowcontrol | blocking entry point only today |
| 5 | "Stage readiness" scheduler state | vllm-omni RFC #4590 checklist | unchecked, unowned |
