# Radix-native session cache — reproducible external test bundle

Test the radix-native session cache (**sgl-project/sglang#27058** + **ai-dynamo/dynamo#10214**)
against a real multi-agent workload **without pi, Node, or any model API key**.

A captured agent trace (`golden-trace-12x4.jsonl`) carries the exact session structure,
token-prefix hashes, and timing of a 12-orchestrator × 4-subagent pi run. The replay driver
reconstructs that KV/token workload against a live Dynamo frontend and toggles the
`nvext.session_control` lifecycle the feature is about — so you reproduce the result with just
the two PRs built + a GPU.

## What the feature does (and the result you'll reproduce)

Under concurrent multi-agent load the KV pool saturates and the radix cache thrashes (LRU evicts
prefixes still in use). With `--enable-session-radix-cache`, each agent session's KV is tagged
and bulk-freed when the session closes (`nvext.session_control {action: close}`), so the pool
never pressures into a forced eviction.

Measured on this hardware (2× L40S 46GB, GLM-4.7-Flash TP2, page-size 16, KV pool = 143,872 tokens),
12 concurrent diverse orchestrators × 4 subagents each:

| Metric | OFF (plain radix) | ON (session radix cache) |
|---|---|---|
| Peak pool occupancy | **100% (saturated, 16 free tokens)** | **66% (held ~34% free)** |
| Forced LRU evictions | **324,768 tokens** | **0** |
| Cache hit rate | 0.999 | 0.998 |

See `kv-pressure-rerun.png` (3-panel: pool occupancy / resident evictable KV / cumulative forced
evictions). The single cleanest contrast: **OFF force-evicts 324K tokens; ON force-evicts 0** —
session close frees KV proactively (`drain_pending_release → allocator.free`, which never touches
the forced-eviction counter).

### This bundle reproduces it (replay-driven, no pi)

Running the steps below — replaying `golden-trace-12x4.jsonl` on/off at `--speedup 4` — reproduces
the same contrast from the trace alone (`kv-pressure-replay.png`):

| Arm | Peak pool occ | Resident cached KV | Forced LRU evictions |
|---|---|---|---|
| `--mode off` | 100% (16 free) | 100% (143,584) | **344,608** |
| `--mode on` | 99% transient | **29% (42,352)** | **2,048** (≈0, 168× fewer) |

This is a *cleaner* A/B than the pi run: both arms replay the identical workload, so the only
variable is `session_control`. (ON shows a tiny 2,048 rather than 0 only because `--speedup 4`
packs the timeline tighter than the original wall-clock; `--speedup 1` reproduces 0. The mechanism
is unmistakable in panel 2: ON's resident session cache stays at 29% because closes free it, while
OFF pins 100%.)

## Contents

| File | What |
|---|---|
| `golden-trace-12x4.jsonl` | Captured `dynamo.agent.trace.v1` trace: 69 sessions, 358 turns, ~2M input tokens, 447s span |
| `replay_session_trace.py` | Replay driver via the **Dynamo** OpenAI path (`nvext.session_control`) |
| `run_agg_radix.sh` | Launch Dynamo frontend + 1 SGLang worker, radix-native ON |
| `replay_sglang_native.py` | Replay driver via **raw SGLang** (`/generate` + `/close_session`) — no Dynamo |
| `run_sglang_native.sh` | Launch a single `sglang.launch_server`, radix-native ON — no Dynamo |
| `kv-pressure-rerun.png` | A/B graph from the live pi run (provenance) |
| `kv-pressure-replay.png` | A/B reproduced by the Dynamo-path replay driver |
| `kv-pressure-sglang-native.png` | A/B reproduced by the **no-Dynamo** raw-SGLang replay |

## Prerequisites

1. Build the two PRs into one venv:
   - sglang#27058 (branch with `--enable-session-radix-cache`): `uv pip install -e python`
   - dynamo#10214: `cd lib/bindings/python && maturin develop --release --uv && cd ../../.. && uv pip install -e .`
2. A GPU box that fits your model. The trace was captured on GLM-4.7-Flash (MoE, ~60GB BF16 → TP2
   on 2×46GB). Any model works for the *mechanics*; the exact KV numbers above assume this setup.

## Reproduce — Option A: via Dynamo's OpenAI endpoint

```bash
# 1. Launch the aggregated server with the feature on (frontend :8000, file discovery, no etcd/NATS)
DYNAMO_DIR=/path/to/dynamo ./run_agg_radix.sh            # TP2, GLM-4.7-Flash, --enable-session-radix-cache

# 2a. ON arm — replay the trace WITH session_control (open/close lifecycle)
python3 replay_session_trace.py golden-trace-12x4.jsonl --url http://localhost:8000 --mode on --speedup 4

# 2b. OFF arm — identical workload, session_control stripped (the baseline)
python3 replay_session_trace.py golden-trace-12x4.jsonl --url http://localhost:8000 --mode off --speedup 4
```

Watch the worker KV gauges on `:8081/metrics` (or `--enable-metrics`):
`sglang:token_usage`, `sglang:kv_available_tokens`, and the forced-eviction counter
`sglang:evicted_tokens_total{cache_type="RadixCache"}`. OFF saturates and the eviction counter
climbs; ON stays bounded and the counter stays at 0.

For a clean A/B, restart the worker between arms (cold cache), or run OFF then ON on fresh servers.
`--speedup` compresses the 447s timeline to create pressure (4 ≈ saturates OFF on a 143K-token pool);
`--speedup 1` reproduces the original wall-clock.

## Reproduce — Option B: no Dynamo at all (single SGLang worker)

The feature is entirely SGLang-side (sglang#27058); Dynamo#10214 is just plumbing that turns
`nvext.session_control` into SGLang's native session calls. So you can test it against one raw
`sglang.launch_server` — no Dynamo, no router, no NATS:

```bash
# 1. One SGLang worker with the feature on (native /generate + /close_session on :30000)
./run_sglang_native.sh                                   # GLM-4.7-Flash TP2
# small box? any model works (the trace sends synthetic token ids):
#   MODEL=Qwen/Qwen3-0.6B TP=1 GPU=0 MEM_FRACTION=0.3 ./run_sglang_native.sh   # add --vocab to the replay

# 2. ON arm — tag each request with session_params.id, /close_session per session
python3 replay_sglang_native.py golden-trace-12x4.jsonl --mode on  --speedup 4
# 3. OFF arm — identical workload, no session_params (plain radix)
python3 replay_sglang_native.py golden-trace-12x4.jsonl --mode off --speedup 4
```

Native mechanism (no Dynamo concepts): a request carries `session_params={"id": sid}` →
`_tag_session_leaf` stamps the leaf (gated by `--enable-session-radix-cache`); `POST /close_session
{session_id}` → `release_session` bulk-frees that session's KV (the worker logs
`release_session <id>: freed N nodes`). Watch `:30000/metrics`:
`sglang:token_usage`, `sglang:kv_available_tokens`, `sglang:evicted_tokens_total{cache_type="RadixCache"}`.

This reproduces the same contrast through pure SGLang — in fact the cleanest version of it
(`kv-pressure-sglang-native.png`):

| Arm (raw SGLang, no Dynamo) | Peak pool occ | Resident cached KV | Forced LRU evictions |
|---|---|---|---|
| `--mode off` | 100% (16 free) | 99% (142,800) | **344,608** |
| `--mode on` | 98% transient | **27% (39,376)** | **0** |

(ON hits exactly 0 here — the native `/close_session` releases each session's KV promptly, so the
pool never pressures into a forced eviction. The worker log shows `release_session <id>: freed N
nodes` for all 69 sessions.) For tiny models pass `--vocab <vocab_size>` to the replay so synthesized
token ids stay in range (deterministic, so prefix reuse is preserved).

## How the replay is faithful

- **Token synthesis is byte-identical to the Dynamo mocker** (`lib/mocker/src/loadgen/trace.rs`):
  `token_id = hash & 0xFFFFFFFF`, each `input_sequence_hashes` entry expanded to `trace_block_size`
  copies, truncated to `input_length`, sent as an IntegerArray `prompt` to `/v1/completions`
  (pre-tokenized passthrough, `--skip-tokenizer-init`). The worker sees the *same block hashes*
  across a session's turns, so the radix cache hits exactly as the real run did (verified: replay
  reaches 0.999 cache-hit rate).
- **Session lifecycle mirrors pi-dynamo-provider**: first turn carries `session_control.action=open`,
  later turns the bare `session_id`, and a throwaway `max_tokens=1` request carries `action=close`
  after the session's last turn.
- **Concurrency is reproduced**: sessions start staggered at their real relative arrival offsets and
  keep their real inter-turn gaps (closed-loop within a session).

## Provenance

Captured from a live pi run: 12 pi orchestrators (round-robin over 6 task themes — Mars landing logs,
history eyewitness accounts, bestiary entries, GTM briefs, city guides, science explainers), each
delegating to 4 `worker` subagents in parallel via pi-subagents, routed through pi-dynamo-provider
(`DYN_AGENT_TRACE=1`) to Dynamo's frontend, with the server-side `dynamo.agent.trace.v1` JSONL sink
enabled. The ON arm is the trace here; the OFF baseline is generated by the replay's `--mode off`.
