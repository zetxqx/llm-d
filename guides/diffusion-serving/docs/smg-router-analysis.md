# SMG (Shepherd Model Gateway): KV-cache routing, metrics dependency, omni support

**Analyzed:** `smg/` clone at commit `8155b1fd2` (main, 2026-07-03), lightseekorg.
Rust routing gateway descended from the SGLang-router design (same `cache_aware`
algorithm and `balance_abs/rel_threshold` config vocabulary; imports
`sglang_scheduler.proto` for gRPC worker compat), rebuilt as an engine-agnostic
gateway (vLLM, SGLang, TensorRT-LLM, TokenSpeed, Ollama + cloud providers).
All paths relative to the smg repo root.

Three questions answered here:

1. [How does the cache-aware (KV) routing work?](#1-cache-aware-routing-a-three-tier-mechanism)
2. [Does routing depend on model-server metrics?](#2-does-routing-need-model-server-metrics)
3. [Is there vLLM-Omni / diffusion / TTS support?](#3-vllm-omni--media-generation-support-none)

**TL;DR:** cache routing is a three-tier mechanism (real KV events over gRPC →
approximate token tree → approximate string tree) with a shortest-queue
override on imbalance. Load signals are a deliberate hybrid: router-local
in-flight counters (fast, always on) + engine metrics polled every 60 s
(vLLM `/metrics`, SGLang `/v1/loads`) + an in-flight-since-last-poll delta
that corrects staleness between polls. **No omni support at all**:
`/v1/images/generations`, `/v1/audio/speech`, `/v1/videos` are unrouted and
hit a 404 fallback.

---

## 1. Cache-aware routing: a three-tier mechanism

Policy implementation: `model_gateway/src/policies/cache_aware.rs` (the design
comment at lines 1–42 is an accurate spec). Ten policies exist
(`policies/factory.rs:94-111`: random, round_robin, passthrough, power_of_two,
least_load, cache_aware, bucket, manual, consistent_hashing, prefix_hash);
default is round_robin; `cache_aware` is the flagship.

The tier is selected per worker group by connection mode and KV-event
availability (`has_event_indexer()`, `cache_aware.rs:877-883`):

### Tier 1 — Event-driven (gRPC workers publishing KV events) — *real* cache state

- Each gRPC worker gets a background task opening a server-streaming
  `SubscribeKvEvents` RPC (`worker/kv_event_monitor.rs:101-191`), receiving
  `KvBlocksStored` / `KvBlocksRemoved` batches with sequence numbers
  (reconnect with 100 ms–30 s exponential backoff).
- Events feed a per-model `PositionalIndexer`
  (`crates/kv_index/src/event_tree.rs`): a `DashMap<(position, ContentHash),
  SeqEntry>` where `ContentHash` = XXH3 of the block's token ids. Incoming
  request tokens are chunked into full blocks
  (`compute_request_content_hashes`, `event_tree.rs:163-173`; partial trailing
  block discarded — matching what engines actually cache) and matched with a
  jump-search (`find_matches`, default stride 64).
- Selection (`score_overlap()`, `cache_aware.rs:928-983`): among workers with
  overlap > 0, max by tuple **(overlap_score DESC, load ASC, tree_size ASC)**;
  no overlap anywhere → fall back to min-load.

This is the Dynamo-KV-router-style event-sourced index — ground truth pushed
from the engine, never scraped.

### Tier 2 — Approximate token tree (gRPC, no KV events)

Router-side radix tree over token ids (`crates/kv_index/src/token_tree.rs`),
**built from the router's own routing history** — the classic sglang-router
approximation: "I sent this prefix to worker X, so X probably has it cached."
Page-aligned (16 tokens/page, mirroring SGLang's radix cache granularity).
Decision (`select_worker_with_tokens`, `cache_aware.rs:986-1073`): compute
`match_rate = matched/input tokens`; if `match_rate > cache_threshold`
(default 0.5) route to the matched tenant, else route to min-load and insert
the prefix there.

### Tier 3 — Approximate string tree (HTTP workers)

Identical algorithm on raw characters (`select_worker_with_text`,
`cache_aware.rs:1076-1155`; `crates/kv_index/src/string_tree.rs`) — the router
never tokenizes on the HTTP hot path. (On the gRPC path, `input_ids` arrive
pre-tokenized: `routers/grpc/common/stages/worker_selection.rs:161`. The HTTP
router passes extracted chat text: `routers/http/router.rs:197`.)

### The load override: imbalance detection

Cache affinity is ignored whenever the pool is imbalanced
(`is_imbalanced()`, `cache_aware.rs:283-307`) — then it's shortest-queue.
Three OR'd triggers:

1. **Count spread** (router-local in-flight counters): `(max_load − min_load)
   > balance_abs_threshold` (default 32) **AND** `max_load > min_load ×
   balance_rel_threshold` (default 1.1).
2. **KV spread** (engine-polled): `max_token_usage − min_token_usage >
   balance_token_usage_threshold` (default 1.0 = disabled).
3. **KV overload ceiling** (engine-polled): `max_token_usage >
   overload_token_usage_threshold` (default 1.0 = disabled).

Note the defaults: out of the box, only the router-local count spread is
active; the KV-utilization triggers are opt-in.

### Housekeeping

- **Eviction:** background task every `eviction_interval_secs` (30 s), LRU per
  tenant, `max_tree_size` 10k nodes per tree (`cache_aware.rs:149-222`).
- **Worker add/remove:** add inserts an empty prefix for the worker;
  remove is a deliberate no-op — LRU eviction ages out stale tenants
  (`cache_aware.rs:368-424`).
- **Multi-router replicas:** a gossip **mesh** replicates tree inserts across
  router instances (`mesh/adapters/tree_sync.rs`): `TreeDelta = (tree_kind,
  node_hash, worker_url, epoch)` broadcast per model, with a paged
  repair-stream protocol for hashes a peer doesn't know. This is SMG's answer
  to the ledger-drift-across-replicas problem (Dynamo's equivalent:
  `replica_sync`).
- **P/D disaggregation:** independent prefill/decode policy instances per
  model (`policies/registry.rs:346-385`), sticky-key namespaced per leg;
  vLLM (sequential gRPC + KV transfer) and SGLang (parallel HTTP + bootstrap)
  variants; docs at `docs/concepts/routing/pd-disaggregation.md`.

## 2. Does routing need model-server metrics?

**It runs without them, but its best load signals are polled from the
engine.** SMG is a textbook fast-ledger + slow-metrics hybrid:

### Router-local (always on, per-request freshness)

- Atomic in-flight counter per worker: increment on dispatch, decrement on
  completion (`worker/worker.rs:822,893-903`), read in a single O(workers)
  snapshot pass (`RoutingState`, `worker.rs:785-819`).
- Used by: cache_aware's count-spread trigger and load tie-breaks, and as the
  universal fallback for power_of_two / least_load when no engine snapshot
  exists.

### Engine-polled (background monitor, 60 s default)

`WorkerMonitor` (`worker/monitor.rs:474-605`), one polling loop per
`(model, worker_type, connection_mode)` group (`group_monitor_loop`,
`monitor.rs:778-922`) — and it **skips polling entirely if no load-aware
policy is active for the group** (`monitor.rs:793`). Engine-specific fetch
(this is commit `8155b1fd2`, PR #1867):

| Engine | Endpoint | Data |
| --- | --- | --- |
| vLLM | `/metrics` (Prometheus text) | `vllm:num_requests_running/waiting`, `vllm:gpu_cache_usage_perc` (v0) / `vllm:kv_cache_usage_perc` (v1) — **ratio only, no absolute tokens** |
| SGLang | `/v1/loads`, fallback `/metrics` | running/queued reqs, `token_usage`, `gen_throughput`, `cache_hit_rate`, `num_waiting_uncached_tokens` (absolute token work, `/v1/loads` only) |
| gRPC workers | `GetLoads` RPC | full `WorkerLoadResponse` incl. absolute tokens, DP-rank breakdown |

Consumers:

- **`least_load`** — the most sophisticated policy: score ≈
  `(queued_tokens + inflight_tokens) / throughput + kv_pressure_weight ×
  k/(1−k)`. Config: `kv_pressure_weight` 0.15, `mean_prefill_tokens` 1024,
  `default_throughput` 2000 tok/s (`config/types.rs:373-396`).
- **`power_of_two`** — compares `effective_token_usage()` of two sampled
  candidates; falls back to local request counts when snapshots are missing.
- **`cache_aware`** — KV spread/overload triggers only (via a `watch` channel
  the monitor broadcasts into, `monitor.rs:911-916`).

### The staleness correction (the notable design detail)

`least_load` tracks tokens it dispatched **since the last poll** in a local
`inflight_tokens` map (add on selection, `least_load.rs:249-252`; reset on
each fresh snapshot, `:269-280`). So the polled snapshot provides ground
truth, and the router's own dispatch log water-fills the 60-second gap —
exactly the fix for the herding/double-booking failure mode of scrape-only
routing.

### What is *not* a routing input

- Health checks (`/health` probe, 60 s, 3-fail/2-success thresholds,
  `worker/manager.rs:123-475`) and circuit breakers gate *eligibility*, not
  scoring.
- The "40+ Prometheus metrics" SMG exports are self-observability; with
  `--engine-metrics` it also re-exports scraped engine gauges as
  `smg_engine_*` — observability only, no feedback into routing.

### Placed against the other two routers

| | llm-d (EPP) | Dynamo (KV router) | SMG |
| --- | --- | --- | --- |
| Primary load signal | scraped vLLM metrics (+ EPP-side active-request-scorer) | router-side predictive ledger (token blocks) | router-local counters + polled engine load |
| Staleness handling | scrape interval | ledger is per-dispatch | in-flight-since-poll delta corrects the 60 s scrape |
| Cache state | prefix-cache scorer (approx) / events | KV events → indexer | KV events (gRPC) → indexer, else approximate trees |
| Multi-replica routers | stateless scorers tolerate it | `replica_sync` | gossip mesh tree-sync |

SMG is the closest shipping implementation of the layered design argued in
[routing-opportunities.md](routing-opportunities.md): fast local ledger for
per-request decisions, slow polled metrics for calibration/imbalance, push
events for cache truth.

## 3. vLLM-Omni / media generation support: none

**Verdict: no support, and no partial path.** Evidence:

- **Route table** (`model_gateway/src/server.rs:771-824`): chat/completions,
  embeddings, rerank, responses/conversations, Anthropic `/v1/messages`,
  Gemini interactions, classify, tokenize, realtime WS/WebRTC sessions, and
  `/v1/audio/transcriptions` (speech-**to-text** input, multipart). There is
  **no** `/v1/images/generations`, `/v1/audio/speech`, or `/v1/videos`.
- **Fallback:** unmatched paths hit `sink_handler` → **404**
  (`server.rs:97,965`). Worse than llm-d's EPP (parser 400, fixable with
  `payload-agnostic.yaml`) — there is no passthrough to workers at all.
- Zero grep hits for `diffusion`, `vllm-omni`, `DiT`, `images/generations`,
  `audio/speech` in code or docs. The only "omni" string is gpt-4**o**
  name-matching in cloud-provider discovery
  (`workflow/steps/external/discover_models.rs`).
- The `crates/multimodal` crate is VLM **input** preprocessing (LLaVA-family
  image resize/normalize/tokenize, optional OpenCV frame extraction) — nothing
  about generation.
- Image generation appears only as an **OpenAI MCP tool passthrough**
  (`routers/openai/mcp/tool_handler.rs`) — recording `image_generation_call`
  results from OpenAI's cloud backend, not serving anything.

## Takeaways for the llm-d × vLLM-Omni investigation

1. **The gap holds across all three routers.** llm-d (parser 400 by default),
   Dynamo (routes omni but round-robin), SMG (404) — none has body-aware or
   cost-aware routing for generation workloads. SMG has the most mature
   hybrid *load* model of the three, and it is still denominated entirely in
   tokens and text prefixes.
2. **`least_load` is Tier 2 in token clothing.** `(queued_work +
   in-flight_work) / throughput + pressure` is exactly the declared-cost
   scorer shape — substitute `denoise_steps × resolution_scaling` for token
   counts and vLLM-Omni stage saturation for KV pressure, and this formula is
   the diffusion work-remaining scorer proposed in Tier 2. For diffusion the
   numerator is *more* accurate than for LLMs (deterministic cost).
3. **Two mechanisms worth borrowing regardless of modality:** the
   in-flight-since-last-poll correction (fixes scrape staleness with ~20 lines
   of state), and the tiered cache strategy (real events when the engine
   publishes them, approximate router-side tree when it doesn't — the right
   posture for vLLM-Omni, which publishes no KV events today).
4. **Competitive framing:** SMG validates that "engine-agnostic Rust gateway
   with hybrid load signals" is where routers are converging — but omni
   modalities are outside its scope entirely, so it is not a competitor on
   this seam, it's another proof the seam is unclaimed.
