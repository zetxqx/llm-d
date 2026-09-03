# Router-layer close_session experiment plan

> **2026-09-03 revision, after the engine-level results** (see [close-session-ab-20260902/RESULTS.md](close-session-ab-20260902/RESULTS.md)): two weka-replay rounds (idle cap 1s and 30s) are both null, and a synthetic inversion demo is a clean positive control (8/8 residents survive with close vs 0/8 without, 35x probe TTFT). The gate below therefore fires: the fleet-scale weka A/B (R-close/R-blind/R-noclose performance comparison) is CANCELLED - the workload never enters the inversion regime, so no routing accuracy can produce a cache delta. What replaces it: (1) the fleet capacity sweep (`run_router_capacity_sweep.py`, c=16/32/48/64 through the EPP, close on) - it feeds the admission-control direction RESULTS.md points to, and yields fleet-scale close-delivery accuracy and binding stability as a byproduct; (2) the **inversion demo through the router**: rerun `run_inversion_demo.py` via the EPP under R-close vs R-blind plugin configs. In the inversion regime, delivery accuracy translates directly into survival: R-close (~100% delivery) should reproduce 8/8 residents surviving; R-blind (~25% chance delivery) should lose most residents. This is the one router experiment expected to show a performance difference, and it doubles as the end-to-end proof for the close-routing PR. The sections below are kept for the original arm/measurement design details that both replacements reuse.

**Question**: what does the router (EPP) contribute to close_session? The engine-level A/B (single pod, direct IP) measures whether close helps *when it reaches the owning pod*. Fleet-wide, a close POST through a plain load balancer lands on an arbitrary pod: with 4 replicas, 3/4 of closes are silent no-ops (the target pod has no such session, dereferences nothing). The router's job is delivery: extract the session id from the close body and route it to the bound endpoint. This experiment measures that delivery value in isolation.

## Gate: run this only after the engine-level result is positive

The router can only deliver a benefit the engine provides. If the current single-pod run (idle_cap=30, max_wait_ms=30000) is still negative, the fleet experiment's cache metrics will be flat regardless of routing, and only Phase 1 (correctness/delivery accuracy) is worth running. Decide after RESULTS of the 30s-idle run.

## Mechanism under test

- inference-perf close POST carries only body `{"session_id": ...}` - no session header (verified in openai_client.py). The EPP must parse the body: that is the `session-id-producer` body-field source (llm-d-router `feat/session-control-2003`, commit 16e6f2d1) plus `sessionbinding/close.go` (commit 0c8397d5) which routes a client-initiated close to the bound endpoint.
- Generation requests carry `x-session-id` header + body `session_id`; the session-binding tracker binds session -> endpoint on first request; the session-control filter pins subsequent requests.
- The currently deployed EPP image (`zmq-metrics-v1`) does NOT have close routing - a new image must be built from `feat/session-control-2003` (rebase onto synced main first; `cloud-build-epp` skill).

## Arms

All arms run through the EPP (standalone mode, envoy sidecar, fleet of 4 sglang pods), all with identical session-affinity config. Only close handling differs. Same image everywhere; arms toggled via the plugins ConfigMap (established A/B/C pattern: helm upgrade + `kubectl rollout restart deployment/session-control-epp`).

| Arm | Client sends close? | EPP close routing | Expected |
|---|---|---|---|
| R-close | yes | on (close.go routes to bound pod) | full engine benefit, fleet-wide |
| R-blind | yes | off (close falls through to normal scheduling, lands on an effectively arbitrary pod) | ~25% of closes land right by chance: ~1/4 of the R-close benefit |
| R-noclose | no | n/a | baseline; R-blind should sit between R-noclose and R-close, close to R-noclose |

The R-close vs R-blind delta is the router's contribution; R-blind vs R-noclose isolates the chance-delivery floor. If R-blind ~= R-close, the router layer adds nothing (e.g. eviction pressure too low, or closes arrive too late to matter).

## Workload

- Fleet capacity: 4 pods x 2.34M pool; knee expected around 48-64 concurrent sessions. Use `--sessions 48` minimum, 64 preferred.
- Traces: the 60MB slice has only 39 traces - insufficient. Download a ~250MB slice (~100+ complete traces). Do NOT use `duplicate_sessions_target`: duplicated sessions share hash_ids, creating cross-session KV sharing that inflates hit ratios and breaks capacity realism.
- Timing: `trace_idle_gap_cap_seconds: 30` + `max_wait_ms: 30000` (both, same block). Stage timeout 4800-6000s per arm.
- Arms run SEQUENTIALLY (they share the fleet). Between arms: `flush_cache` on all 4 pods, restart EPP with the arm's plugins config, verify active config from the EPP startup log (`grep config-file`).
- Client: in-cluster Job (same pattern), but `base_url` = EPP service (`http://<epp-clusterip>:80`), not a pod IP.

## Measurements

Primary (from inference-perf per-request reports, same pipeline as make_charts.py):
- Fleet-wide main-agent cache-hit mean/p10, TTFT p50/p90, prefix-loss events per 5-min bucket, split pre/post pool-fill.

Mechanistic (the router-specific evidence - collect even if primary metrics are flat):
- **Close delivery accuracy**: envoy access log (per-request upstream pod attribution, `envoy-access-log.values.yaml`) - for each close POST, did it land on the pod that served that session's generation requests? Report % correct per arm (R-close should be ~100%, R-blind ~25%).
- **Effective closes per pod**: sglang log lines `release_session <id>: indexed N component leaves` - N>0 means the close dereferenced something real; N=0 on a pod means a misdelivered no-op. The per-pod ratio of N>0 closes is the ground truth of delivery, independent of the access log.
- **Binding stability**: from access logs, each session's generation requests should hit exactly one pod for the whole run (affinity sanity check; a session that migrates poisons both its own hit ratio and the close target).

## Failure modes to watch

- Spot pod restart mid-arm: bindings point at a dead endpoint; sessions rebind and their KV restarts cold on the new pod. Record pod restarts; if any arm loses a pod, rerun that arm (fleet arms are sequential, so a restart biases one arm only).
- EPP ConfigMap staleness: EPP reads plugins config at startup only - every arm switch REQUIRES the rollout restart, and the restart wipes the binding table (acceptable: arms flush caches anyway; do the restart BEFORE the arm starts).
- `/close_session` path through envoy: standalone mode forwards all paths through ext-proc; verify in Phase 1 that the close POST actually reaches the EPP filter (access log + EPP debug log) and is not short-circuited.
- HF tokenizer download in the Job: unchanged from single-pod runs (HF_TOKEN secret optional).

## Phases

1. **Build + smoke (0.5 day)**: rebase `feat/session-control-2003` on main, run its unit tests, build image via cloud-build-epp, deploy with R-close plugins config. Smoke: one short trace through EPP; verify (a) all turns of a session hit one pod, (b) the close POST lands on that same pod (access log), (c) that pod logs `release_session ... indexed N>0`, other pods log nothing.
2. **Fleet A/B/C (1 day)**: three sequential arms x 80-100 min, order R-noclose -> R-blind -> R-close (ascending expected benefit, so any drift across arms works against the hypothesis rather than for it).
3. **Analysis**: extend make_charts.py with the delivery-accuracy table; same chart set per arm plus a 3-way ECDF overlay.

## Success criteria

- Phase 1: delivery accuracy ~100% in R-close smoke. This alone validates the #2003 close-routing code path end-to-end.
- Phase 2: R-close reproduces the single-pod close benefit at fleet scale; R-blind captures only ~1/4 of it. If R-close shows benefit but R-blind ~= R-noclose, the router's close routing is demonstrably necessary, not just sufficient.

## Beyond delivery: what else the router can do with close

Delivering the close to the bound pod is the router's minimal, passive role. The router's real asset is global state (session -> pod bindings, per-session lifecycle, per-pod load), which enables the following, ordered by value/effort. Note items 1, 2 and 4 do NOT depend on the eviction-ordering benefit this A/B measures - they are worth pursuing even if the engine-level result stays negative.

### 1. Router-initiated close (highest practical value)

Real clients (Claude Code included) do not call `/close_session` today; a close protocol that depends on client cooperation has no adoption path. The router can make close an infrastructure policy instead:

- **Idle-TTL reaping**: the binding table already tracks last-activity per session; on TTL expiry (e.g. 10 min idle), the router sends `/close_session` to the bound pod. Zero client changes. A wrong guess costs one soft re-prefill, never correctness.
- **Ref-hygiene argument**: `session_ref` only discriminates if dead sessions actually get closed. On a fleet running for days with no closes, every block ends up referenced, protection degenerates to plain LRU (exactly the degeneration the single-pod noclose arm exhibits). Router TTL-close is what keeps the reference system meaningful - a value independent of eviction-ordering benefits.

### 2. Live-KV ledger for KV-pressure-aware placement (highest performance ceiling)

From response `usage` plus open/close events, the router can maintain an accurate per-pod ledger of live (unclosed) session KV. New sessions then route to the pod with the most free live-KV headroom - a placement signal neither queue depth nor prefix-affinity provides. Without close events the ledger only grows and is useless. Integration point: a live-session-KV scorer alongside the existing queue/kv/prefix scorers.

### 3. Cascade close for subagents (cheap add-on)

The router sees the subagent session-id pattern (`<parent>::sa:<agent>:s<n>`). Two inference rules: the parent's next turn arriving implies the previous subagent batch is dead (close them); a parent close cascades to all its subagent sessions. Clients then manage only one session lifecycle.

### 4. Migration and double-residency hygiene (spot fleets)

When rebalancing a session off a hot pod: unbind, let it re-prefill on the new pod, and immediately close the copy on the old pod - collapsing the double-residency window from "whenever LRU gets to it" to zero. Same for stale copies left behind by binding drift (pod restarts, capacity spill): the router knows the binding history; nobody else knows those dead copies exist.

### 5. Graded eviction priority instead of binary close (research direction)

Close is priority-zero. The richer form is router-injected priority scores at session open (SLA tier, predicted session heat) consumed by the engine's eviction policy - the "router-initialized signals" direction from sglang RFC #29099 / #27574, not yet landed upstream. If the eviction-ordering benefit materializes, this is the natural upstream narrative: close routing first, graded signals as the endgame.

### Protocol note: who can know a session is over, and when

The agent-harness loop (send -> read response -> continue iff tool calls) means "is this the last request" is unknowable at send time, which caps what each close mechanism can cover:

- **Known at send time** (what `x-session-final` on the request requires): only forced-final turns - a subagent's wrap-up request with tools disabled, max-turn cutoffs, pre-compaction summaries. Replay benchmarks can always pre-mark finals because the full script is known in advance - a mild optimism vs live clients that should be remembered when reading results. `x-session-final` is a #2003 router-wire-protocol convention only; sglang never sees it and only understands `/close_session`.
- **Known after the final response** (the workhorse): the harness learns the loop ended when a response carries no tool calls / the subagent returns. It can then POST `/close_session` explicitly - this matches the close timing inference-perf actually benchmarks, at the cost of one extra call.
- **Never known** (TTL's domain): a main-agent session ends when the user walks away; there is no post-hoc moment either (harness exit hooks help but do not survive kill -9). Only infrastructure-side idle-TTL covers this.

Design consequence for #2003: the protocol should read "clients call close once they learn the session ended; infrastructure TTL covers the rest", with the final-request header demoted to an optional optimization for forced-final turns.
