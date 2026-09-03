# Benchmark Log — Session-Control Weka Replay

Append-only lab log for the benchmark campaign described in [`../benchmark-setup.md`](../benchmark-setup.md). One entry per action, written at the time the action is taken. Never edit past entries; append corrections as new entries.

**Rules for a valid recorded run:** (1) the reproducibility manifest below is current and the guide's git SHA is recorded; (2) the active router arm is confirmed from the EPP startup log, not assumed; (3) spot preemption events during the run window are checked — a preempted decode pod invalidates the run; (4) every knob that differs from the manifest is recorded in the run entry.

## Reproducibility Manifest

Everything needed to rebuild this exact stack. Update this section (with a log entry) whenever any value changes.

| Component | Pinned value |
|---|---|
| Cluster | GKE `bobbm`, project `bobzetian-gke-dev`, us-central1, context `gke_bobzetian-gke-dev_us-central1_bobbm` |
| Namespace | `llm-d-session-control` |
| GPU nodes | `bobbm-spoth100` spot pool, `nvidia-h100-80gb`, 4 GPUs/node (⚠ spot: preemptions must be checked per run) |
| Fleet | 4 replicas × TP2 (8× H100), `strategy: Recreate`, aggregated (no P/D) |
| SGLang image | `lmsysorg/sglang:v0.5.15.post1` = `sha256:00c53fe4c31bf22d7b37537f28bbdfd924c02de13cdfb4bff7378c9c34d75ab2` |
| Engine flags | `--context-length 262144 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.88 --enable-session-radix-cache --radix-eviction-policy priority --enable-cache-report` (full args in [`patch-sglang.yaml`](../modelserver/gpu/sglang/base/patch-sglang.yaml)) |
| Measured KV pool | `max_total_num_tokens = 2,342,038` per replica (from `/get_server_info`, 2026-08-06) |
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` @ HF revision `dcaee4d4dfc5ee71ad501f01f530e5652438fde0` (captured from harness tokenizer fetch, 2026-08-06) |
| Router chart | `llm-d-router-standalone-v0.9.0`, helm release `session-control` (rev 1, installed 2026-07-16) |
| EPP image | `ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:873179822ab0895a37ea09f2112ca39a6ae50a26612561c8bfad7f9a8c5af6f5` |
| Envoy sidecar | `envoyproxy/envoy@sha256:85500e28ed088ec39ff0adc1be3d358a8ad062926aaa62c36b28bde00919e4e8` |
| Router arms | A [`session-control.values.yaml`](../router/session-control.values.yaml) · B [`optimized-baseline.values.yaml`](../router/optimized-baseline.values.yaml) · C [`scorer-only.values.yaml`](../router/scorer-only.values.yaml); every switch = `helm upgrade` + `kubectl rollout restart deploy/session-control-epp` + startup-log confirmation |
| Router values stack | base → [`monitoring-gmp.values.yaml`](../router/monitoring-gmp.values.yaml) → [`envoy-access-log.values.yaml`](../router/envoy-access-log.values.yaml) → arm file, ALL FOUR on every upgrade (helm replaces values wholesale). Release at revision 6 (2026-08-07) |
| Observability | GMP PodMonitoring ([`monitoring.yaml`](monitoring.yaml)): decode `:8000` + EPP `:9090` (auth off). Per-request (session→pod) attribution: Envoy HCM access log JSON on the envoy-proxy container stdout |
| Harness | inference-perf in-cluster Job ([`inference-perf-weka.yaml`](inference-perf-weka.yaml)), custom image `us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/inference-perf@sha256:bae74353b6efb4062120a7a1d6ddc78abd7303093877f667e8e28d2b9f6cb01b` (digest-pinned; stock `quay.io/...@sha256:9a93ed9d...` was used for smoke + c16) |
| Harness source ref | kubernetes-sigs/inference-perf `f6c714b` + fork branch `weka-context-window-clamp` @ `0c82d74` (context_window_clamp; upstream-PR candidate) |
| Replay determinism | `base_seed=1786064268311` (pinned; = arm-B c16's seed) · `trace_idle_gap_cap_seconds=30.0` · `context_window_clamp=258048` (from c32 onward) — identical across all arms/stages |
| Corpus | `semianalysisai/cc-traces-weka-with-subagents-060826-256k` @ snapshot `74105717d89542231d66e2495872b49c5176b339` (Apache 2.0) |
| Replay tokenizer | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (must equal served model) |
| Think-time cap | smoke 5.0s / recorded runs 30.0s — must be identical across compared arms |
| Guide git state | `0bab6861` on branch `sessioncontrol` of the local llm-d clone (local-only, not pushed). Note: the unrelated pr27058-verify microbench rework remains uncommitted and is excluded from this SHA |

**Known caveats affecting interpretation:** EPP approximate prefix index matches at most 131,072 prefix tokens (~2/3 of the median 195K prefix) — relevant to arms B/C; inference-perf does not echo `x-session-token`, so arm A's pinning does not engage from this harness until the token-echo patch lands; sglang#29173 (upstream 2026-08-02) changes session-radix semantics — do not compare runs across an image upgrade over that boundary.

## Campaign Checklist

- [x] Model selection + capacity sizing (Qwen3-Coder-30B-A3B-FP8, 4×TP2) — see [`../benchmark-setup.md`](../benchmark-setup.md)
- [x] Fleet redeployed and verified (2026-08-06)
- [x] Harness manifest ready (`inference-perf-weka.yaml`)
- [x] Commit guide state; record SHA in manifest (`0bab6861`, 2026-08-06)
- [ ] Switch EPP to arm B (optimized-baseline) + confirm via startup log
- [x] Smoke test: c4 × 8 sessions against arm B — PASSED (2026-08-06; 7/8 sessions, 0.1% request errors, root-caused; see run entry)
- [x] Prometheus scraping (2026-08-07): GMP `PodMonitoring` for decode `:8000` + EPP `:9090` ([`monitoring.yaml`](monitoring.yaml)); EPP metrics auth disabled via [`../router/monitoring-gmp.values.yaml`](../router/monitoring-gmp.values.yaml); both targets verified `up` on the GMP collectors
- [x] Per-request pod attribution (2026-08-07): Envoy HCM access log now emits per-request JSON with `%UPSTREAM_HOST%` + `%REQ(X-SESSION-ID)%` ([`../router/envoy-access-log.values.yaml`](../router/envoy-access-log.values.yaml)); verified: `{"upstream":"10.100.9.8:8000","session":"probe-A",...}`. Rejected alternatives: EPP `/debug/plugins/state` (aggregate only), EPP metrics (per-pod gauges only)
- [x] GCS report storage (2026-08-07): bucket `gs://bobzetian-session-control-bench` created, in-cluster write verified, harness config now uploads to `reports/{timestamp}`; harness image pinned by digest in the Job spec
- [x] Calibrate `peakPrefillThroughput` (2026-08-07): **68,228 tok/s** measured (CHUNK_SIZE=8192 = engine chunked_prefill_size, 20 samples via EPP) vs plugin default 15,928 (4.3× underestimate avoided); set in arm-B values, active in EPP (revision 7, confirmed in startup log)
- [ ] Arm B sweep: c16 → c32 → c48 → c64 → c72 → c80, one Job per stage (clean validity boundaries: spot-preemption check + counter snapshots between stages). No separate warmup pass — replayed sessions are unique, so cold-start is the workload's real semantics; comparability comes from identical procedure per arm (EPP restart → same stage sequence)
- [ ] Arm C sweep (same stages)
- [ ] Token-echo patch for inference-perf; then Arm A sweep
- [ ] Engine A/B on arm A: session-radix on vs off
- [ ] Analysis + writeup in `benchmark-results/`

## Run Entry Template

```
### <YYYY-MM-DD HH:MM TZ> — <arm> <stage> <smoke|recorded>
- Arm confirmed: <EPP startup-log line: config file + plugin list>
- EPP pod: <name>; decode pods: <names> (all Ready before start: y/n)
- Harness: job <name>, image digest <sha256:...>, config: stages=<...>, idle_gap_cap=<s>, num_sessions=<n>
- Warmup: <done how / skipped because>
- Window: <start> → <end>; spot preemptions in window: <none | details → run INVALID>
- Per-pod counter snapshots: BEFORE <num_requests_total / prompt_tokens_total per pod> → AFTER <same>; deltas = this run's distribution
- Results: <throughput, TTFT p50/p90, ITL p50, cache-hit % (cached_tokens), per-pod balance>
- Report location: <job log / GCS path>
- Anomalies / deviations from manifest: <...>
```

## Log

### 2026-08-05 — Planning
- Model selected: Qwen3-Coder-30B-A3B-Instruct-FP8 (criteria and rejected alternatives in [`../benchmark-setup.md`](../benchmark-setup.md)). Weka corpus + inference-perf chosen as harness.

### 2026-08-06 — Fleet redeploy (Qwen3-32B → Qwen3-Coder-30B-A3B-FP8, 3→4 replicas)
- Old deployment deleted (model label is in the immutable selector; delete+apply required). New manifests applied with `strategy: Recreate` (spot pool too full for rolling surge).
- 2 pods scheduled immediately on freed GPUs; autoscaler added node `jr29` (pool 4→5 nodes); all 4 Ready ~10 min after apply.
- Verified: `/get_server_info` → context 262144, kv fp8_e4m3, session_radix=True, `max_total_num_tokens=2,342,038`; completions route through EPP; `x-session-token` pinned 4/4 turns to pod `sw4z6`.

### 2026-08-06 — EPP restart (config-reload discipline established)
- Learned: EPP reads the plugins ConfigMap at startup only; helm upgrade alone does not roll the pod. All arm switches now require explicit `rollout restart` + startup-log confirmation (docs updated).
- Restarted EPP; startup log confirmed **arm A active** (`session-control-plugins.yaml`: session-affinity-filter + queue/kv-util/prefix scorers 2/2/3). Noted from same log: approximate-prefix producer `maxPrefixTokensToMatch=131072`, blockSize forced 16→64.

### 2026-08-06 — Harness prepared
- Local inference-perf synced to kubernetes-sigs upstream `f6c714b` (today); `upstream` remote added, main tracks upstream/main; fork remains `origin` for patches.
- Confirmed weka replay support (`trace_session_replay` + `weka_trace_replay` datagen, subagent dependency graph, think-time cap, `session_id_header_key`). Confirmed gap: no response-header echo → arm A pinning inert from this harness until patched.
- In-cluster Job manifest written: [`inference-perf-weka.yaml`](inference-perf-weka.yaml) (smoke shape c4×8, 5s cap).
- **Current state: fleet idle on arm A; next action = switch to arm B + smoke test.**

### 2026-08-06 — Guide state committed
- Benchmark setup committed as `0bab6861` (branch `sessioncontrol`, local llm-d clone, commit only — not pushed): modelserver manifests, README, benchmark-setup.md, benchmark/ (LOG.md + harness Job), router arm B/C values files. The pre-existing pr27058-verify microbench rework was deliberately excluded (separate work, still uncommitted).
- Note: LOG.md itself is inside the committed tree — subsequent log entries make the working tree dirty relative to `0bab6861`; that is expected. Re-commit the log periodically; config files are what the manifest SHA guarantees.

### 2026-08-06 ~17:25 PDT — Switched to arm B (optimized-baseline)
- `helm upgrade` release `session-control` → revision 2, chart pinned `v0.9.0` (env.sh floats `v0`; pinned deliberately), values: base + `optimized-baseline.values.yaml`. Anonymous registry config workaround used for the OCI pull.
- EPP rollout-restarted; startup log confirms `config-file=/config/optimized-baseline-plugins.yaml`, plugins: approx-prefix-cache-producer, inflight-load-producer, prefix-cache-affinity-filter, token-load-scorer. **Arm B active.**
- ⚠ `peakPrefillThroughput` NOT yet calibrated (running plugin default 15,928 tok/s, tuned for dense Qwen3-32B) — acceptable for smoke, must calibrate before recorded runs.

### 2026-08-06 ~17:27 PDT — Smoke test started (arm B, c4 × 8 sessions)
- Job `inference-perf-weka` applied (smoke shape: concurrent_sessions=4, num_sessions=8, idle_gap_cap=5.0s, request_timeout=600).
- Harness image digest captured: `sha256:9a93ed9d...` (see manifest). Image supports `weka_trace_replay` — no fallback build needed.
- Runtime pins observed in logs: model revision `dcaee4d4...`, corpus snapshot `7410571...` (both recorded in manifest). Benign warning: corpus-text tokenization length notice from transformers (prompt-synthesis path, not a request).
- Decode fleet all 4 Ready at start; awaiting completion → results will be appended.

### 2026-08-06 ~17:57 PDT — FALSE START: first smoke run produced zero traffic for 30 min
- Symptom: Job Running 30 min, zero requests at EPP/decode pods; harness pod pegged at exactly 1 core, RSS ~10GB and climbing; last log line = trace download.
- Root cause: `num_dataset_entries: 500` — the weka datagen compiles ALL configured entries into execution graphs with fully synthesized prompt text, single-threaded, BEFORE replay starts, even though only `num_sessions: 8` are replayed. 500 entries ≈ hours of preprocessing.
- Fix: `num_dataset_entries: 12` (just above num_sessions); Job deleted and recreated ~17:58. Compile took ~90s (3184 events / 12 sessions), replay started 17:59:56 PDT (00:59:56 UTC). Lesson recorded as a comment in `inference-perf-weka.yaml`.
- Verdict after source review (inference-perf `f6c714b`): not a correctness bug — a missing optimization. `WekaTraceReplayDataGenerator` unconditionally uses the eager path (`_build_sessions_from_traces` → `initialize_sessions`, weka_trace_replay_datagen.py:645), compiling every loaded entry (~7.5s/entry single-threaded, prompts held in RAM) regardless of `num_sessions`. The framework already ships a lazy path (`initialize_sessions_lazy` / `_build_session_for_slot`, replay_graph_session_datagen.py:1249) and OTel replay uses it (otel_trace_replay_datagen.py:469); weka simply never adopted it. Upstream issue/PR candidate; until then keep `num_dataset_entries` ≈ sessions needed (large sweeps: expect ~7.5s × entries of Job startup).

### 2026-08-06 ~18:00–18:50 PDT — Smoke run #2 (arm B, c4×8) in progress
- Traffic confirmed: 877+ requests / 82M+ prompt tokens across all 4 pods; skewed distribution (sw4z6 ~47%) consistent with prefix-affinity routing. Observed a live turn with `#cached-token: 104,003 / #new-token: 330` (99.7% prefix hit). Token usage ~0.04 (no pressure at c4, as intended).
- One session failed mid-replay (`wekatrace5`, 18:20 PDT, 74 downstream events cancelled). No errors in SGLang/EPP/Envoy logs → client-side (timeout or parse); awaiting per-request error detail in the final report. Other 7 sessions healthy.

### 2026-08-06 18:55 PDT — Smoke run #2 COMPLETED (arm B, c4 × 8 sessions) — PASSED
- Arm confirmed: EPP startup log `config-file=/config/optimized-baseline-plugins.yaml` (approx-prefix-cache-producer, inflight-load-producer, prefix-cache-affinity-filter, token-load-scorer). ⚠ `peakPrefillThroughput` uncalibrated (default 15,928) — smoke only, do not treat numbers as recorded baseline.
- EPP pod `7d6747cc9-949vp`; decode pods `nwt52/sw4z6/t96nl/vhkww`, all Ready at start.
- Harness: job `inference-perf-weka`, image `sha256:9a93ed9d...`, stages=[c4×8sessions], idle_gap_cap=5.0s, num_dataset_entries=12, request_timeout=600.
- Window: 17:59:56 → 18:55 PDT (stage duration 3320s). **Spot preemptions: none (0 restarts on all 4 decode pods) → run VALID.**
- Results (stage 0): 8 sessions → 7 succeeded / 1 failed; 1029 events → 954 completed / 74 cancelled / 1 error (0.1%). Fleet input throughput 27,719 tok/s; output 187 tok/s. **TTFT med 313ms / p90 725ms at prompt med 94,655 tokens** → near-total prefix reuse (a cold 95K prefill would be seconds); corroborated by live scheduler line `#cached-token: 104,003 / #new-token: 330`. ITL med 5.9ms, TPOT ~6ms. Request latency med 2.34s. Session duration med 359s / max 3213s (one 692-event session dominated wall clock).
- Per-pod request distribution (final): sw4z6 502 (52%), t96nl 182, nwt52 148, vhkww 130 — strong skew, consistent with prefix-affinity routing at zero memory pressure. Watch this at c48+: this is exactly the affinity-vs-imbalance failure mode ThunderAgent §3.2 documents.
- `sglang:cache_hit_rate` gauge read 0.0 post-run — known stale-while-idle gauge gotcha; do not use it, derive hit rate from `cached_tokens` in responses / TTFT.
- **wekatrace5 failure root cause (from report `by_label`): `400 - Context Window` — input 260,127 + requested completion 8,630 = 268,757 > 262,144.** The `-256k` corpus filters INPUT ≤256K but input+max_tokens can exceed the window; server rejects, harness cancels the session's remaining 74 events (12.5% session failure from one tail request). Second upstream-issue candidate: weka replay should clamp max_tokens to (context − input), configurably.
- Report location: job log (structured stage JSON + tables); local files in pod at `reports-20260807-005748` (pod gone — for future runs enable GCS storage or extract before completion).

### 2026-08-07 16:59 UTC — Arm B RECORDED stage c16 STARTED
- Arm confirmed: revision 7, `optimized-baseline-plugins.yaml` with `peakPrefillThroughput:68228` (startup log).
- Harness: c16 × 32 sessions, num_dataset_entries=36, idle_gap_cap=30.0s (recorded-run value), image digest-pinned, GCS reports → `gs://bobzetian-session-control-bench/reports/{timestamp}`.
- BEFORE counter snapshot (16:59:14Z): nwt52 153/10.45M · sw4z6 509/49.87M · t96nl 192/17.70M · vhkww 139/14.69M (reqs / prompt toks).
- Attribution: envoy access-log JSON live on envoy-proxy stdout (session→upstream per request).
- Results to be appended on completion.

### 2026-08-08 05:19 UTC — Arm B RECORDED stage c48 STARTED
- c48 × 96 sessions, entries=100, clamp corrected to **255,000** (see c32 finding), same pinned seed/cap/image. BEFORE counters = c32 AFTER (fleet idle in between).

### 2026-08-08 ~16:20 UTC — c48 mid-run observations (still running, ~11h in)
- Not stuck: all 96 sessions dispatched; engines at token usage ~0.02 with 0-1 running requests each — the tail-drain phase where the few longest sessions (400-700 events × ≤30s think gaps) run serially. Wall clock of a stage ≈ longest session chain, NOT throughput; GPUs are idle by design (think-time realism is what keeps KV resident between turns).
- ⚠ 14 sessions failed so far — clamp 255,000 breached again: observed drift **2.87%** (input recorded ~250K → actual 257,129; overflow by 43 tokens). Drift budget must rise: set `context_window_clamp: 251000` (≈4.2% budget) BEFORE c64.
- ⚠ Harness pod at 24.8GB / 32Gi limit — raise Job memory limit to 48Gi before c64 (132 eager-compiled entries + retained state).

### 2026-08-08 ~05:10 UTC — Arm B stage c32 COMPLETED — VALID, clamp margin finding
- Window: 2026-08-07 22:11 → 2026-08-08 ~05:10 UTC (~7h). Spot preemptions: none (0 restarts) → VALID.
- AFTER counters (05:17:05Z): nwt52 3280/313.60M · sw4z6 2926/264.09M · t96nl 3320/341.90M · vhkww 2929/269.61M. **Deltas: 2098 / 1621 / 1891 / 1987 (27.6/21.3/24.9/26.2%)**, total 7597 = report successes exactly; 711.7M prompt tokens.
- Results: input throughput 52,277 tok/s, output 586 tok/s. TTFT mean/med/p90 = 756 / 434 / 1396 ms; ITL med 7.1ms / p90 15.1ms; TPOT med 7.6ms. Prompt med 81,052 / p90 184,406; output med 452. Session duration med 4682s / p90 8672s.
- Sessions: 52/64 succeeded (18.8% failed); 10,128 events → 7,597 completed / 2,519 cancelled. Errors: **12× `400 Context Window`** — the 258,048 clamp was ACTIVE but insufficient.
- **Finding 3 — retokenization drift is proportional, not fixed.** Failing shape: clamped completion 1,559 + recorded input ~256.5K, but the server tokenized the synthesized text + chat template to 261,255 input tokens (**+1.86%**). A fixed 4,096 margin cannot cover proportional drift at the input ceiling; bound: clamp ≤ window − drift×max_input ≈ 257,000. Fixed by config only (no rebuild): `context_window_clamp: 255000` (tolerates ≤2.79% drift, 1.5× the observed). c32's completed-request metrics remain comparable; its session-failure rate shares c16's caveat.
- Note for the upstream PR: document that the clamp must be sized proportionally below the serving window (retokenization + chat-template drift), not window − small constant.
- Harness image switched to the fork build: `us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/inference-perf@sha256:bae74353...` (= upstream `f6c714b` + branch `weka-context-window-clamp` @ `0c82d74`; built with Cloud Build — API had to be enabled in the project first; note the corp account cannot stream build logs, but builds succeed — check `gcloud builds list` not the submit output).
- Config: c32 × 64 sessions, entries=68, idle_gap_cap=30.0s, `base_seed=1786064268311` (pinned), `context_window_clamp=258048` (first stage WITH the clamp). Config validated by the patched image (field accepted).
- BEFORE counters (22:10:38Z) = c16's AFTER values (no traffic in between): nwt52 1182/107.66M · sw4z6 1305/136.28M · t96nl 1429/147.18M · vhkww 942/86.35M.
- Arm B confirmed still active (revision 7, no EPP changes since calibration).

### 2026-08-07 20:25 UTC — Arm B stage c16 COMPLETED — VALID, with findings
- Window: 16:59:20 → ~20:24 UTC (~3.4h). Spot preemptions: none (0 restarts all pods) → VALID.
- AFTER counter snapshot (20:25:18Z): nwt52 1182/107.66M · sw4z6 1305/136.28M · t96nl 1429/147.18M · vhkww 942/86.35M. **Deltas: 1029 / 796 / 1237 / 803 reqs (27/21/32/21%) — well balanced**, total 3865 = report's success count exactly; 385M prompt tokens replayed.
- Results: input throughput 32,819 tok/s, output 331 tok/s. TTFT mean/med/p90 = 713 / 390 / 1233 ms; ITL med 6.3ms; TPOT med 6.7ms. Prompt med 86,040 / p90 201,646; output med 431. Session duration med 1573s / p90 4611s / max 9562s.
- Sessions: 26/32 succeeded (**18.8% failed**); 4326 events → 3865 completed / 455 cancelled. Errors: **5× `400 Context Window`** + 1× `503 Service Unavailable` (0.2% of requests, amplified by cascade cancellation).
- Reports uploaded to `gs://bobzetian-session-control-bench/reports/` (per-timestamp prefix); envoy access log holds per-request (session→pod) records for stickiness analysis.
- **Finding 1 — cascade amplification worsening with scale** (1 error-killed session at c4 → 5 at c16): patched the fork — `SessionReplayConfig.context_window_clamp` clamps max_tokens to (clamp − recorded prompt tokens); branch `weka-context-window-clamp` @ `0c82d74`, custom image building via Cloud Build → `us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference/inference-perf:weka-clamp-0c82d74`. Config will set clamp=258048 (262,144 − 4,096 margin for chat-template overhead — observed server-side count ran ~4K over recorded input).
- **Finding 2 — base_seed was timestamp-derived** (`1786064268311` this run): session shuffle + token synthesis differ per run, breaking cross-arm comparability. Fixed: `load.base_seed` pinned to **1786064268311** (this run's seed) in the harness config, keeping the completed c16 in the comparable set. All future stages/arms replay identical schedules.
- ⚠ Comparability note: c16 ran WITHOUT the clamp; c32+ will run WITH it. The clamp only affects requests whose prompt+completion exceeded the window (the ones that previously 400'd), so completed-request metrics remain comparable; session success rates are not comparable between c16 and later stages. Re-run c16 with the clamp at campaign end if per-stage session-failure curves are needed.

### 2026-08-07 — Report persistence + calibration (checklist items 3, 4)
- GCS: bucket `gs://bobzetian-session-control-bench` (us-central1); in-cluster `gsutil cp` write test passed on node SA credentials; harness `storage.google_cloud_storage` enabled (`reports/{timestamp}`); harness image pinned to digest `sha256:9a93ed9d...` in the Job spec.
- Calibration: `guides/recipes/router/calibration/calibrate.sh` with GUIDE_NAME=session-control, CHUNK_SIZE=8192 (matched to live `chunked_prefill_size` from `/get_server_info`) → **peakPrefillThroughput = 68,228 tok/s**. Written into `optimized-baseline.values.yaml` plugin parameters; helm revision 7; EPP restart; startup log confirms `peakPrefillThroughput":68228` under `optimized-baseline-plugins.yaml`. Re-run calibration after any model/TP/GPU/chunk-size change.

### 2026-08-07 morning — Observability infrastructure (checklist items: Prometheus, pod attribution)
- GMP scraping live: PodMonitoring for decode `:8000` and EPP `:9090` (`benchmark/monitoring.yaml`), both targets `up` on collectors. EPP metrics auth off via new `router/monitoring-gmp.values.yaml` (chart's ServiceMonitor feature unusable — cluster runs GMP, not prometheus-operator; first upgrade attempt failed on the missing CRD and was rolled back cleanly).
- Per-request (session→pod) attribution for arms B/C: `router/envoy-access-log.values.yaml` overrides the chart v0.9.0 envoy bootstrap, replacing the 8081 HttpConnectionManager access-log placeholder format with JSON incl. `%UPSTREAM_HOST%` + `%REQ(X-SESSION-ID)%`. Two false starts recorded for honesty: (1) first edit landed on the listener-level logger (L4 — HTTP fields empty), (2) second edit hit the listener-level occurrence again because both loggers share the same placeholder string; third edit targets the second occurrence (HCM) and verified end-to-end.
- Helm release now revision 6; values stack is base + monitoring-gmp + envoy-access-log + arm (all four required on every upgrade — README updated). Arm B confirmed active after each restart.
- EPP restarts during this work wiped router in-memory state several times — irrelevant now (no recorded runs in progress), but during the campaign, infra changes must not happen mid-stage.
