# Micro-trace: hero agent vs. churn — the smallest close_session demo

`microtrace_hero_churn.py` is the minimal workload that shows the benefit of explicit `/close_session` **per request** instead of as pool aggregates: one long-lived agent's turns stay fast when dead sessions are closed, and go bimodal when they aren't.

## The design

- **Hero**: one agent session with a big context (default ~8k tokens) that takes a turn every 25 s (think time) for the whole run and is never closed — it hasn't ended. Every turn logs `prompt_tokens`, `cached_tokens`, hit % and latency to a CSV.
- **Churn**: short-lived sessions (default ~3k tokens, 3 turns) arriving continuously. In `--mode on` each is tagged with `session_id` and closed when it ends; in `--mode off` the identical traffic runs with no session lifecycle.
- **The mechanism under test**: in OFF mode churn residue accumulates until the pool is full, and LRU then evicts the *oldest* entries — which is precisely the hero's prefix, idle since its last turn 25 s ago. In ON mode churn KV is freed at close, the pool never fills, and the hero stays cached.

## Server prep: shrink the pool

At the stack's default pool (321,614 tokens) you would need huge churn volumes; cap the pool instead so pressure arrives in about a minute. Add to the args in `guides/session-control/modelserver/gpu/sglang/base/patch-sglang.yaml`:

```yaml
            - "--max-total-tokens=40000"
```

then apply and re-scale (rollout takes ~10 min/pod plus H100 spot waits):

```bash
kubectl --context $CTX apply -n $NS -k ${REPO_ROOT}/guides/session-control/modelserver/gpu/sglang/gke/
kubectl --context $CTX -n $NS scale deployment/session-control-gpu-sglang-decode --replicas=2
kubectl --context $CTX -n $NS rollout status deployment/session-control-gpu-sglang-decode --timeout=40m
```

Sizing at 40k: hero ≈ 10k tokens (25%), in-flight churn ≈ 8k, so ON mode has ample headroom, while OFF-mode residue (~3.7k dead tokens per churn session at 8/min) fills the remaining pool in roughly a minute and then every new churn prefill forces evictions. **Revert the line and re-apply when done** — a 40k pool cripples normal use of the stack.

## Run the A/B

Both arms need `--enable-session-radix-cache`, `--radix-eviction-policy=priority` and `--enable-cache-report` (all already in the guide's base config). Run each arm on its own pod, or sequentially on one pod with a `flush_cache` between:

```bash
k cp microtrace_hero_churn.py <pod>:/tmp/microtrace_hero_churn.py
k exec <pod> -- curl -s -X POST localhost:8000/flush_cache
k exec <ON-pod>  -- python3 /tmp/microtrace_hero_churn.py --url http://127.0.0.1:8000 --mode on  --out /tmp/hero_on.csv
k exec <OFF-pod> -- python3 /tmp/microtrace_hero_churn.py --url http://127.0.0.1:8000 --mode off --out /tmp/hero_off.csv
k cp <ON-pod>:/tmp/hero_on.csv hero_on.csv && k cp <OFF-pod>:/tmp/hero_off.csv hero_off.csv
```

Defaults run 5 minutes (~12 hero turns, ~40 churn sessions). For reruns on a warm pod use `--salt <new>` (disjoint traffic) and `--sid-suffix=_<new>` (closed session ids are tombstoned and can never re-tag).

## Reading the result

Each run prints a per-turn line and a summary; the CSVs have `t, turn, prompt_tokens, cached_tokens, hit_pct, latency_s`. Expected picture (turn 1 is a cold prefill in both arms — exclude it):

- **ON**: hit rate stays ~97% every turn, latency flat (~1 s at 8k context on this stack).
- **OFF**: once the pool fills (~1 min in), some hero turns drop to ~0% cached and pay a full re-prefill — latency goes bimodal, and the summary's `min hit` / `max latency` capture the spikes.

The single-sentence takeaway the demo produces: *the same agent doing the same work gets predictable turn latency only when dead sessions around it are closed.*

## Knobs

| flag | default | meaning |
|---|---|---|
| `--duration` | 300 | run length (s) |
| `--hero-context` / `--hero-turn-growth` | 8000 / 200 | hero context size and per-turn growth (tokens) |
| `--hero-interval` | 25 | hero think time (s) — the LRU vulnerability window |
| `--churn-context` / `--churn-turns` | 3000 / 3 | dead-residue volume per churn session |
| `--churn-rate` | 8 | churn sessions per minute — pressure dial |
| `--salt` / `--sid-suffix` | "" | rerun hygiene (disjoint tokens / fresh session ids) |

Content is deterministic hex words at ~8 tokens/word on the Qwen3 tokenizer (`TOKENS_PER_WORD` in the script — recalibrate for other tokenizers).
