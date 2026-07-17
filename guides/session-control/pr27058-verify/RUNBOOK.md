# Runbook — how the PR 27058 A/B benchmark was run

Exact steps to reproduce the A/B verification of sglang#27058 (session radix cache) on the GKE stack. Results and analysis are in [RESULTS-gke-qwen3.md](RESULTS-gke-qwen3.md); the graph is `figures/kv-pressure-ab.png`.

This folder lives inside the session-control guide at `guides/session-control/pr27058-verify/`; run steps 3–8 from this directory (they reference local files). All commands assume:

```bash
export REPO_ROOT=$(git rev-parse --show-toplevel)   # root of the llm-d clone
export CTX=gke_bobzetian-gke-dev_us-central1_bobbm
export NS=llm-d-session-control
alias k="kubectl --context $CTX -n $NS"
```

## 1. Upgrade the stack to an image that has the feature

The feature first shipped in sglang v0.5.15 (the PR merged 2026-06-23; v0.5.13.post1/v0.5.14 predate it). The session-control guide's modelserver base already carries the two required config changes:

- `guides/session-control/modelserver/gpu/sglang/base/kustomization.yaml` — an `images:` override pinning `docker.io/lmsysorg/sglang` to `v0.5.15.post1` (the recipes component pins v0.5.13.post1).
- `guides/session-control/modelserver/gpu/sglang/base/patch-sglang.yaml` — the two server args: `--radix-eviction-policy=priority` and `--enable-session-radix-cache`.

Apply and wait (each pod needs image pull + Qwen3-32B TP2 load, ~10 min; H100 spot pool may need a scale-up — one zone failed with "GCE out of resources" and the autoscaler retried another):

```bash
kubectl --context $CTX apply -n $NS -k ${REPO_ROOT}/guides/session-control/modelserver/gpu/sglang/gke/
k rollout status deployment/session-control-gpu-sglang-decode --timeout=40m
```

Sanity-check the flag is live:

```bash
k exec <pod> -- curl -s localhost:8000/get_server_info | python3 -c "import json,sys; i=json.load(sys.stdin); print(i['enable_session_radix_cache'], i['radix_eviction_policy'], i['version'])"
# expect: True priority 0.5.15.post1
```

## 2. Scale to two pods (one per arm)

```bash
k scale deployment/session-control-gpu-sglang-decode --replicas=2
```

Arm assignment in this run: pod `…-hzztk` = **ON** (replay sends session ids + `/close_session`), pod `…-svznv` = **OFF** (identical workload, no session lifecycle). Both pods run identical server flags — the arms differ only in what the client sends.

## 3. Stage the workload in the pods

`kubectl port-forward` collapses under the replay's 69 concurrent threads (requests hang, metrics time out) — run everything **inside** the pods instead:

```bash
for pod in <ON-pod> <OFF-pod>; do
  k cp replay_sglang_native.py $pod:/tmp/replay_sglang_native.py
  k cp golden-trace-12x4.jsonl $pod:/tmp/golden-trace-12x4.jsonl
done
```

`replay_sglang_native.py` here is patched relative to the gist (see § Caveats): released-API `session_id` field, `ignore_eos`, and a `--sid-suffix` flag.

### Alternative: replay through /v1/chat/completions

`replay_sglang_chat.py` runs the same A/B through the OpenAI-compatible chat endpoint instead of native `/generate` — sessions are marked with the same top-level `session_id` field, which SGLang's `ChatCompletionRequest` accepts as an extension (an OpenAI SDK passes it via `extra_body={"session_id": sid}`); release is still `POST /close_session`. Differences from the native replay:

- The trace's token hashes are rendered as deterministic hex words (the server tokenizes and applies the chat template), so prefix sharing is preserved but token counts are approximate (`--tokens-per-word`, default 4). Read exact counts from the aggregated `usage` the script prints.
- No `--vocab`; use `--salt <wave>` to make a pass's traffic disjoint from a previous wave (pass 1: default salt, pass 2: `--salt r2`), and `--sid-suffix` for fresh session ids exactly as with the native script.
- `usage.prompt_tokens_details.cached_tokens` is only populated when the server runs with `--enable-cache-report` (off by default upstream; this guide's `base/patch-sglang.yaml` now enables it). Without the flag the field is `null` and the script prints "cached n/a" — judge cache behavior from `sglang:kv_evictable_tokens` and the `release_session` log lines instead.

```bash
k cp replay_sglang_chat.py <pod>:/tmp/replay_sglang_chat.py     # both pods
k exec <ON-pod>  -- python3 /tmp/replay_sglang_chat.py /tmp/golden-trace-12x4.jsonl --url http://127.0.0.1:8000 --mode on  --speedup 4
k exec <OFF-pod> -- python3 /tmp/replay_sglang_chat.py /tmp/golden-trace-12x4.jsonl --url http://127.0.0.1:8000 --mode off --speedup 4
```

Smoke-tested on this stack (2 sessions, 15 turns): all requests 200, and each close logged `release_session <sid>_...: indexed N leaves, freed M nodes` — tagging and release behave identically to the native path.

## 4. Start the metrics samplers (one per pod, before the replay)

Each sampler is a single `kubectl exec` streaming a 1 Hz CSV to the local machine — run each in the background and leave it running across the pass:

```bash
k exec <pod> -- sh -c 'echo "t,token_usage,kv_available,kv_evictable,evicted_total"; while true; do m=$(curl -s -m 5 localhost:8000/metrics 2>/dev/null); tu=$(echo "$m" | awk "/^sglang:token_usage{/{print \$2}"); av=$(echo "$m" | awk "/^sglang:kv_available_tokens{/{print \$2}"); ev=$(echo "$m" | awk "/^sglang:kv_evictable_tokens{/{print \$2}"); fe=$(echo "$m" | awk "/^sglang:evicted_tokens_total{/{s+=\$2} END{print s+0}"); echo "$(date +%s),${tu:-NA},${av:-NA},${ev:-NA},${fe:-0}"; sleep 1; done' > metrics_<arm>.csv
```

## 5. Pass 1 — clean caches, first wave

Flush both pods, then start both replays at the same time (each takes ~217 s at speedup 4):

```bash
k exec <pod> -- curl -s -X POST localhost:8000/flush_cache            # both pods
k exec <ON-pod>  -- python3 /tmp/replay_sglang_native.py /tmp/golden-trace-12x4.jsonl --url http://127.0.0.1:8000 --mode on  --speedup 4 --vocab 151000
k exec <OFF-pod> -- python3 /tmp/replay_sglang_native.py /tmp/golden-trace-12x4.jsonl --url http://127.0.0.1:8000 --mode off --speedup 4 --vocab 151000
```

`--vocab 151000` folds the trace's synthetic 32-bit token hashes into Qwen3's id range (vocab 151,936; staying under 151,643 avoids special-token ids). Expect each run to end `358 turns (358 ok, 0 err)` with `69 /close_session` on the ON arm only.

## 6. Pass 2 — no flush, second wave with fresh token ids

Do **not** flush. Change two things versus pass 1:

- `--vocab 149000` — a different modulus produces disjoint token ids, i.e. a genuinely new wave of traffic (same modulus would be a 100% cache hit on the OFF pod's residue and show nothing).
- `--sid-suffix=_r2` on the ON arm — closed session ids are tombstoned server-side and silently never re-tag KV, so reused ids turn the ON arm into an OFF arm (this exact mistake produced `metrics_on2.csv`; kept as evidence). Note the `=` — argparse eats a space-separated value starting with `-`.

```bash
k exec <ON-pod>  -- python3 /tmp/replay_sglang_native.py /tmp/golden-trace-12x4.jsonl --url http://127.0.0.1:8000 --mode on  --speedup 4 --vocab 149000 --sid-suffix=_r2
k exec <OFF-pod> -- python3 /tmp/replay_sglang_native.py /tmp/golden-trace-12x4.jsonl --url http://127.0.0.1:8000 --mode off --speedup 4 --vocab 149000
```

If the ON pod accumulated residue from a botched pass, `POST /flush_cache` it and redo its pass 2 (that is what `metrics_on3.csv` is).

## 7. Collect and check

Stop the samplers. Confirm the ON arm actually released — the worker logs one line per close and TP rank:

```bash
k logs <ON-pod> --since=10m | grep release_session
# healthy:  release_session <sid>: indexed 84 leaves, freed 129 nodes
# broken:   release_session <sid>: indexed 0 leaves, freed 0 nodes   <- id tombstoned or session_params used
```

Summary stats per CSV (peak/final `kv_evictable`, min `kv_available`, max `evicted_total`) — see the inline python in RESULTS-gke-qwen3.md, or just eyeball the graphs.

## 8. Graphs

```bash
python3 -m venv /tmp/vizenv && /tmp/vizenv/bin/pip install -i https://pypi.org/simple matplotlib seaborn pandas
/tmp/vizenv/bin/python gen_charts.py     # reads metrics_{on,off}.csv + metrics_on3.csv/metrics_off2.csv, writes figures/kv-pressure-ab.png
```

Chart styling follows the [tvhahn/matplotlib-skill](https://github.com/tvhahn/matplotlib-skill) Claude Code skill (installed at `~/.claude/skills/matplotlib`).

(The `-i https://pypi.org/simple` matters on this machine — the default pip index is a private mirror without matplotlib.)

## 9. Cleanup

```bash
k exec <pod> -- curl -s -X POST localhost:8000/flush_cache   # both pods — drop replay junk KV
kubectl --context $CTX apply -n $NS -k ${REPO_ROOT}/guides/session-control/modelserver/gpu/sglang/gke/   # back to replicas: 3 when done with 2-pod experiments
```

## Caveats that will bite a reproducer

1. **Use the released API.** On v0.5.15+ radix-native sessions are the top-level `session_id` field on `/generate`; `session_params={"id": ...}` silently selects the old dedicated-slot session controller (no tagging, close frees nothing). The gist's original scripts predate this rename.
2. **Never reuse a closed session id** (tombstone list, last 8192). Fresh ids per wave — hence `--sid-suffix`.
3. **Don't trust `sglang:kv_evictable_tokens` while the server is idle** — the gauge only refreshes on scheduler activity. Ground truth for "was it freed": resend the prefix without a session id and read `cached_tokens` in `meta_info` (0 = freed).
4. **Don't run the replay through `kubectl port-forward`** — it silently drops most of the 69 concurrent streams.
5. Pool size matters for the eviction signal: this stack's pool (321,614 tokens) is 2.2× the gist's, so pass 1 alone never forces evictions — the OFF-arm thrash only appears in pass 2 once residue occupies the pool.
