# vLLM-Omni inter-stage tensor transfer: edges, keys, and the chunk receive path

The mechanics *under* [vllm-omni-stage-pipeline.md](vllm-omni-stage-pipeline.md) §2:
how a stage knows where to send tensor data, how concurrent requests stay
untangled, and the exact code path by which a consuming stage fetches a chunk
and turns it into model input. All file references into the `vllm-omni/`
clone; the Dynamo contrast references
[dynamo-omni-stage-internals.md](dynamo-omni-stage-internals.md).

## 1. Stages send to *edges*, not addresses

An **edge** is a directed link between two stages in the pipeline graph —
"data flows from stage A to stage B." Stages are the nodes; the arrows are
the edges. Qwen3-Omni has 3 stages and 2 edges:

```
Thinker (0) ──edge (0→1)──► Talker (1) ──edge (1→2)──► Code2Wav (2)
```

In code an edge is literally the tuple `("0", "1")` — the key of the
connector map `{(from_stage, to_stage): connector}`. Most models are chains,
but the general form is a graph: a stage with `input_sources=(0, 1)` in its
`pipeline.py` topology has two incoming edges (e.g. BAGEL's DiT pulling from
the Thinker).

### How edges are configured (deploy YAML)

The `connectors:` section defines named connector instances; each stage binds
them to edges via `input_connectors` / `output_connectors`:

```yaml
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector          # or MooncakeTransferEngineConnector
    extra: { host: 10.0.0.5, zmq_port: 50051 }   # transport params
stages:
  - stage_id: 1
    input_connectors:
      from_stage_0: connector_of_shared_memory   # ← this line IS the wiring
```

The parser (`load_omni_transfer_config`,
`vllm_omni/distributed/omni_connectors/utils/initialization.py:190-300`)
turns key `from_stage_0` on stage 1 into edge `("0", "1")` → `ConnectorSpec`.
Both endpoints may declare the same edge (receiver via `input_connectors`,
sender via `to_stage_N`); a connector-type mismatch is a hard error. Specs
are **role-neutral** — sender/receiver role is injected per process depending
on whether the caller's stage_id is the edge's `from` or `to` side
(`create_connectors_from_config`, `:114-123`). Dynamo synthesizes default SHM
edges when the YAML omits them (`_ensure_stage_connectors` in its
`stage_worker.py`).

### How "where" is resolved, per connector type

| Connector | Addressing |
| --- | --- |
| `SharedMemoryConnector` | none — deterministic `/dev/shm` block names (≤64 KB payloads inlined into the ticket). Same-node assumption IS the address. |
| `MooncakeStoreConnector` | by key in a distributed object store; both sides configured with the store address; neither knows the other's IP |
| `MooncakeTransferEngineConnector` (P2P RDMA) | the only real peer addressing, from the edge's `extra`: sender binds a ZMQ listener at `host:zmq_port`; receiver gets `sender_host`/`sender_zmq_port` (defaulted from the same `host`, `initialization.py:117-118`). Ports are deterministic per edge: `base + purpose offset + from_stage + local_rank × stride` (`:87-112`). |

Per request, the exact data location travels dynamically as the **metadata
ticket returned by `put()`** (SHM block name; Mooncake `source_host`/
`source_port`). The pattern is **publish-and-advertise, then pull** — nobody
pushes tensors at a stage's IP; the consumer redeems the ticket. Consequence:
same-node → cross-node is purely a YAML edit (swap connector type, fill in
hosts); stage code and control-plane protocol don't change.

## 2. Concurrent requests: everything is keyed by request ID

The connector key has three parts (`chunk_transfer_adapter.py:201,304`):

```python
connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"
# "chatcmpl-8f3a2b…_0_17" = request 8f3a2b…, produced by stage 0, chunk #17
```

- **request_id** — unique per request, minted ONCE at admission (API server;
  in Dynamo the router: `str(uuid.uuid4())`), threaded unchanged through
  every stage. Does double duty as correlation key *and* cleanup key
  (evicting a finished/cancelled request's blocks).
- **stage_id** — the producer, so one request's Thinker and Talker outputs
  can't collide.
- **chunk_id** — per-request sequence counter (chunked streaming = many
  payloads per request).

Underneath, connectors namespace by edge too: internal key
`{key}@{from_stage}_{to_stage}` (`connectors/base.py:106-112`).

Consumers don't browse for their data — they **construct** the exact key they
expect next from per-request counters (`get_req_chunk[req_id]`) and poll for
it. 50 interleaved requests = 50 disjoint key families; no ambiguity.

Dynamo's disagg mode is the degenerate case: one payload per stage per
request, key = bare `request_id`, and the consumer only calls `get()` after
receiving the ticket — no speculative polling.

## 3. The receive path, in execution order (code walk)

The two edges of the Qwen3-Omni pipeline carry very different cargo, so both
are walked below: first **Talker → Code2Wav** (discrete codec tokens — the
simpler case, §3 acts 1–5), then **Thinker → Talker** (continuous
embeddings + hidden states — §3b).

Concrete case first: Code2Wav (stage 2) pulling codec chunks from the Talker.
Five acts.

### Act 1 — Park: scheduler sees a request needing its next chunk

`_process_chunk_queue_legacy` (`chunk_transfer_adapter.py:556-584`), called
from inside `schedule()`:

```python
self.load_async(request)                          # → adapter's _pending_load_reqs
request.status = RequestStatus.WAITING_FOR_CHUNK  # parked; not batched
queue.remove(request)
self.requests_origin_status[request.request_id] = target_status  # remember
```

`WAITING_FOR_CHUNK` is an omni-added request status: while set, the vLLM
scheduler simply skips the request, so others keep batching — waiting for
data costs zero GPU.

### Act 2 — Poll: a daemon thread spins on the pending list

`recv_loop` (`transfer_adapter/base.py:49-83`), one per stage engine —
non-blocking poll of every parked request, requeue on miss, 1 ms backoff
(tight-spinning on failed `shm_open` "can burn a full CPU core"):

```python
for _ in range(len(self._pending_load_reqs)):
    request = self._pending_load_reqs.popleft()
    if not self._poll_single_request(request):
        self._pending_load_reqs.append(request)    # not landed yet
```

### Act 3 — Fetch: build the key, ask the connector

`_poll_single_request` (`chunk_transfer_adapter.py:195-215`):

```python
target_stage_id = stage_id - 1                    # my upstream
chunk_id = self.get_req_chunk[req_id]             # per-request counter
connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"
result = self.connector.get(str(target_stage_id), str(stage_id), connector_get_key)
if result is None:
    return False                                  # requeue
self.get_req_chunk[req_id] += 1                   # next poll → chunk_id+1
```

### Act 4 — Use: the payload is written INTO the Request object

Same function. For Code2Wav (`model_mode="generation"`, `:244-280`) the
fetched codec tokens literally become the request's **prompt**:

```python
new_ids = payload_data.get("codes", {}).get("audio")  # Talker's codec tokens
request.prompt_token_ids = new_ids                    # chunk = next stage's input
request.additional_information = info                 # tensors/meta merged here
request.num_computed_tokens = 0                       # reprocess with new prompt
```

Completion is in-band (`meta.finished` → `finished_requests`). Guard at
`:281-287`: an *empty* non-final chunk returns `False` so the stage can't
start before the first real frame lands. Success ends with
`self._finished_load_reqs.add(req_id)`. (The Talker's `model_mode="ar"`
branch of this same function is different enough to get its own section —
§3b below.)

### Act 5 — Wake: next schedule() cycle un-parks it

Back in `_process_chunk_queue_legacy` (`:575-581`):

```python
if request.request_id in finished_load_reqs:
    request.status = target_status                # restore WAITING or RUNNING
    self.requests_with_ready_chunks.add(request.request_id)
```

The request re-enters normal vLLM scheduling carrying the chunk in
`prompt_token_ids`/`additional_information`. Sync points: schedulers call
`_consume_pending_connector_output` at the top of every `schedule()`
(`omni_ar_scheduler.py:217`) and `_capture_omni_connector_output` at the tail
of every `update_from_output()` (`omni_scheduler_mixin.py:115-130`) — chunk
state syncs exactly once per cycle. Safety net: requests waiting too long on
a dead producer are force-failed `FINISHED_ERROR`
(`_process_pending_input_timeouts`, mixin `:76-113`).

The loop repeats — generate against this chunk, run dry, re-park for chunk
N+1 — which is how a Code2Wav request alternates between "waiting 1 ms for
the next codec frame" and "vocoding" while the Talker is still producing,
all inside ordinary continuous batching.

## 3b. The Thinker → Talker edge: hidden states, not tokens

The Talker → Code2Wav edge carries **discrete token IDs** that become the
next stage's prompt. The Thinker → Talker edge carries **continuous
tensors** — embeddings and last-layer hidden states — and those can't be a
prompt. The same park/poll/wake machinery is reused, but both the send and
the use sides differ.

### Send side (Thinker, stage 0)

The producer path is the adapter's save thread: `_send_single_request`
(`chunk_transfer_adapter.py:294-345`) takes the `multimodal_output` the model
runner captured during forward passes and hands it to the model-specific
packaging hook `custom_process_next_stage_input_func` — for Qwen3-Omni that
is `thinker2talker_async_chunk`
(`model_executor/stage_input_processors/qwen3_omni.py:435-510`), whose
docstring is the summary: "thinker's text generation outputs (token IDs +
hidden states) → split hidden states into prompt embeddings + generated
embeddings → package for talker."

What goes into the `OmniPayloadStruct` (everything `.detach().cpu()`):

```python
payload = OmniPayloadStruct(
    embed=EmbeddingsStruct(
        prefill=thinker_emb...,          # thinker text embeddings for the prompt
        tts_bos=..., tts_eos=..., tts_pad=...,  # special TTS control embeddings
    ),
    hidden_states=HiddenStatesStruct(output=thinker_hid...),  # last-layer states
    ids=IdsStruct(all=all_token_ids, prompt=prompt_token_ids),
    meta=MetaStruct(finished=...), speaker=speaker, language=language,
)
```

Two subtleties:

- **Chunk 0 is accumulated, not streamed eagerly** (`:478-506`): the
  processor buffers payloads in `transfer_manager.request_payload`
  (returning `None` = "don't send yet"), concatenating `embed.prefill` and
  `hidden_states.output` across calls until the prefill embeddings cover the
  full prompt — only then does the first `put()` happen. Later chunks go
  through the streaming variant
  (`_construct_thinker2talker_streaming_input_async_chunk`, `:265`).
- **Wire asymmetry is intentional**: the sender writes struct attributes,
  the receiver reads dict keys — `OmniMsgpackDecoder` is type-erased, so the
  payload round-trips struct → dict (comment at `:340-345`; schema changes
  must update both ends, see `test_wire_round_trip`).

Then the generic tail: `connector.put(from="0", to="1",
put_key=f"{req_id}_0_{chunk_id}", data=payload_data)`.

### Receive side (Talker, stage 1) — the `model_mode="ar"` branch

In `_poll_single_request` (`chunk_transfer_adapter.py:226-235`), the AR
branch does NOT touch `prompt_token_ids` with payload content:

```python
request.additional_information = payload_data      # tensors ride here
if chunk_id > 0 and request.resumable:
    construct_next_stage_streaming_input_prompt(payload_data, request)
```

The prompt trick lives in `construct_next_stage_streaming_input_prompt`
(`omni_connectors/adapter.py:219-247`): async-chunk downstream stages are
**prewarmed before the real Talker prompt is known**, so when a new Thinker
segment arrives, the helper extends the request's prompt with **placeholder
zeros** sized from the upstream ids — and refreshes block hashes so the
scheduler allocates KV slots for the extended prompt *without discarding
already-computed state*:

```python
next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
new_prompt = [0] * next_prompt_len          # placeholders, never real content
request.prompt_token_ids.extend(new_prompt)
request.update_block_hashes()               # KV slots for the extension
```

So for this edge the token IDs exist **only to make vLLM's scheduler
allocate KV and batch slots** — the real content enters at forward time: the
model runner reads `additional_information` (per-request state,
`gpu_ar_model_runner.py:386,1492`) and the Talker projects the tensors into
its own hidden dimension via `project_thinker_outputs`
(`models/qwen3_omni/qwen3_omni_moe_talker.py:188-222`) — `text_projection`
for thinker embeddings, `hidden_projection` for hidden states (the two
models have different hidden sizes) — and uses the result as its input
embeddings in place of token-embedding lookup.

### The two edges side by side

| | Thinker → Talker | Talker → Code2Wav |
| --- | --- | --- |
| Cargo | embeddings + hidden states (continuous) | codec token IDs (discrete) |
| Lands in | `request.additional_information` | `request.prompt_token_ids` |
| Prompt tokens | placeholder zeros — KV/slot allocation only | the actual input |
| Packaging | model-specific processor (`thinker2talker_async_chunk`), chunk-0 accumulation | generic path, no processor |
| Consumed by | `project_thinker_outputs` at forward time (dimension projection) | ordinary token-driven forward |

That left column is precisely why **prefix caching breaks on omni AR
stages** (RFC #1184): the Talker's "prompt" is placeholder zeros while its
real input is per-request hidden-state tensors — token-based prefix matching
has nothing truthful to match on. The right column is a normal LLM-shaped
input, which is why the TTS talker→vocoder pipeline could ship prefix
caching enabled.

## 4. Why this matters (routing / competitive)

- This park/poll/wake machinery is what native vLLM-Omni's **streaming
  overlap** is made of (stage N+1 works while stage N produces — the
  `async_chunk` TTFP 2790→655 ms win). Dynamo's stage worker has none of it:
  one blocking `get()` per stage after the upstream fully finishes
  ([dynamo-omni-stage-internals.md](dynamo-omni-stage-internals.md) §6).
- The tensor plane is **topology-configured, not discovery-based** — there is
  no dynamic endpoint resolution. But the ticket mechanism itself is
  replica-agnostic: any process holding the ticket and an edge connector can
  redeem it. That's the property a cluster-level stage router (llm-d Tier 4)
  would exploit to choose *which replica* of stage N+1 pulls a given
  request's data at routing time.
- Request IDs minted once at the front door are the correlation spine of the
  whole pipeline — any routing tier inserted above stages must preserve them
  end-to-end (as Dynamo's router does).
