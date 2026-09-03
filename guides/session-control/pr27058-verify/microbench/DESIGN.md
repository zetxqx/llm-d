# Micro-trace: main session + subagent fan-out — the smallest close_session demo

`microtrace_main_subagents.py` is the minimal workload that shows the benefit of explicit `/close_session` **per request** instead of as pool aggregates. It models the standard agentic fan-out pattern: a long main session delegates work to short-lived subagents and consumes only their results — and its turns stay fast when the finished subagents are closed, or go bimodal when they aren't.

## The design

- **Main session**: one agent session with a big context (default ~8k tokens) that takes a turn every 25 s and is never closed — it hasn't ended. The gap between turns is the time it spends blocked waiting on subagents; that idle window is exactly when its prefix is exposed to LRU. Each turn appends a small amount of new context (default 200 tokens) — the stand-in for the subagent results it folds in — and logs `prompt_tokens`, `cached_tokens`, hit % and latency to a CSV.
- **Subagents**: short-lived sessions (default ~3k tokens of working context, 3 turns) arriving continuously. Each does its work, delivers its result, and ends — after which its entire context is dead KV that no one will ever prefix-match again. In `--mode on` each is tagged with `session_id` and closed when it ends; in `--mode off` the identical traffic runs with no session lifecycle.
- **The mechanism under test**: in OFF mode dead-subagent residue accumulates until the pool is full, and LRU then evicts the *oldest* entries — which is precisely the main session's prefix, idle since its last turn 25 s ago. In ON mode subagent KV is freed at close, the pool never fills, and the main session stays cached.

### Modeling choices (and their caveats)

- **The main session appends fresh tokens, not the literal subagent outputs.** For the radix cache the two are equivalent — a small, previously-unseen suffix on the main session's prefix either way — so the workload uses independent deterministic tokens for simplicity.
- **Subagent contexts are fully disjoint from the main session and from each other.** Real subagents often share a system-prompt or repo-context prefix with the parent; the radix cache shares those prefixes regardless of session lifecycle, so they are never residue. What close_session reclaims is the subagent-*unique* KV — and that is the only part this benchmark generates, which is why disjoint content is the right model rather than a shortcut.

## Server prep: shrink the pool

At the stack's default pool (321,614 tokens) you would need huge subagent volumes; cap the pool instead so pressure arrives in about a minute. Add to the args in `guides/session-control/modelserver/gpu/sglang/base/patch-sglang.yaml`:

```yaml
            - "--max-total-tokens=40000"
```

then apply and re-scale (rollout takes ~10 min/pod plus H100 spot waits):

```bash
kubectl --context $CTX apply -n $NS -k ${REPO_ROOT}/guides/session-control/modelserver/gpu/sglang/gke/
kubectl --context $CTX -n $NS scale deployment/session-control-gpu-sglang-decode --replicas=2
kubectl --context $CTX -n $NS rollout status deployment/session-control-gpu-sglang-decode --timeout=40m
```

Sizing at 40k: main session ≈ 10k tokens (25%), in-flight subagents ≈ 8k, so ON mode has ample headroom, while OFF-mode residue (~3.7k dead tokens per finished subagent at 8/min) fills the remaining pool in roughly a minute and then every new subagent prefill forces evictions. **Revert the line and re-apply when done** — a 40k pool cripples normal use of the stack.

## Run the A/B

Both arms need `--enable-session-radix-cache`, `--radix-eviction-policy=priority` and `--enable-cache-report` (all already in the guide's base config). Run each arm on its own pod, or sequentially on one pod with a `flush_cache` between:

```bash
k cp microtrace_main_subagents.py <pod>:/tmp/microtrace_main_subagents.py
k exec <pod> -- curl -s -X POST localhost:8000/flush_cache
k exec <ON-pod>  -- python3 /tmp/microtrace_main_subagents.py --url http://127.0.0.1:8000 --mode on  --out /tmp/main_on.csv
k exec <OFF-pod> -- python3 /tmp/microtrace_main_subagents.py --url http://127.0.0.1:8000 --mode off --out /tmp/main_off.csv
k cp <ON-pod>:/tmp/main_on.csv main_on.csv && k cp <OFF-pod>:/tmp/main_off.csv main_off.csv
```

Defaults run 5 minutes (~12 main-session turns, ~40 subagents). For reruns on a warm pod use `--salt <new>` (disjoint traffic) and `--sid-suffix=_<new>` (closed session ids are tombstoned and can never re-tag).

## Reading the result

Each run prints a per-turn line and a summary; the CSVs have `t, turn, prompt_tokens, cached_tokens, hit_pct, latency_s`. Expected picture (turn 1 is a cold prefill in both arms — exclude it):

- **ON**: hit rate stays ~97% every turn, latency flat (~1 s at 8k context on this stack).
- **OFF**: once the pool fills (~1 min in), some main-session turns drop to ~0% cached and pay a full re-prefill — latency goes bimodal, and the summary's `min hit` / `max latency` capture the spikes.

The single-sentence takeaway the demo produces: *the same main session doing the same work gets predictable turn latency only when the subagents that finished around it are closed.*

## Knobs

| flag | default | meaning |
|---|---|---|
| `--duration` | 300 | run length (s) |
| `--main-context` / `--main-turn-growth` | 8000 / 200 | main-session context size and per-turn growth (tokens) |
| `--main-interval` | 25 | main-session wait-on-subagents time (s) — the LRU vulnerability window |
| `--subagent-context` / `--subagent-turns` | 3000 / 3 | dead-residue volume per finished subagent |
| `--subagent-rate` | 8 | subagents per minute — pressure dial |
| `--salt` / `--sid-suffix` | "" | rerun hygiene (disjoint tokens / fresh session ids) |

Content is deterministic hex words at ~8 tokens/word on the Qwen3 tokenizer (`TOKENS_PER_WORD` in the script — recalibrate for other tokenizers).
