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
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (HF revision: ⬜ capture from `/get_model_info` at first formal run) |
| Router chart | `llm-d-router-standalone-v0.9.0`, helm release `session-control` (rev 1, installed 2026-07-16) |
| EPP image | `ghcr.io/llm-d/llm-d-router-endpoint-picker@sha256:873179822ab0895a37ea09f2112ca39a6ae50a26612561c8bfad7f9a8c5af6f5` |
| Envoy sidecar | `envoyproxy/envoy@sha256:85500e28ed088ec39ff0adc1be3d358a8ad062926aaa62c36b28bde00919e4e8` |
| Router arms | A [`session-control.values.yaml`](../router/session-control.values.yaml) · B [`optimized-baseline.values.yaml`](../router/optimized-baseline.values.yaml) · C [`scorer-only.values.yaml`](../router/scorer-only.values.yaml); every switch = `helm upgrade` + `kubectl rollout restart deploy/session-control-epp` + startup-log confirmation |
| Harness | inference-perf in-cluster Job ([`inference-perf-weka.yaml`](inference-perf-weka.yaml)), image `quay.io/inference-perf/inference-perf:latest` (⬜ pin digest after first pull — `latest` is mutable) |
| Harness source ref | kubernetes-sigs/inference-perf `f6c714b` (2026-08-06) — local clone synced to this |
| Corpus | `semianalysisai/cc-traces-weka-with-subagents-060826-256k` (date-pinned HF build, Apache 2.0) |
| Replay tokenizer | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` (must equal served model) |
| Think-time cap | smoke 5.0s / recorded runs 30.0s — must be identical across compared arms |
| Guide git state | ⬜ session-control guide changes are UNCOMMITTED in the local llm-d clone (base `9512cb89`) — commit and record the SHA before the first formal run |

**Known caveats affecting interpretation:** EPP approximate prefix index matches at most 131,072 prefix tokens (~2/3 of the median 195K prefix) — relevant to arms B/C; inference-perf does not echo `x-session-token`, so arm A's pinning does not engage from this harness until the token-echo patch lands; sglang#29173 (upstream 2026-08-02) changes session-radix semantics — do not compare runs across an image upgrade over that boundary.

## Campaign Checklist

- [x] Model selection + capacity sizing (Qwen3-Coder-30B-A3B-FP8, 4×TP2) — see [`../benchmark-setup.md`](../benchmark-setup.md)
- [x] Fleet redeployed and verified (2026-08-06)
- [x] Harness manifest ready (`inference-perf-weka.yaml`)
- [ ] Commit guide state; record SHA in manifest
- [ ] Switch EPP to arm B (optimized-baseline) + confirm via startup log
- [ ] Smoke test: c4 × 8 sessions against arm B; pin harness image digest
- [ ] Calibrate `peakPrefillThroughput` for Qwen3-Coder-30B-A3B (`guides/recipes/router/calibration`); update arm-B values + manifest
- [ ] Arm B sweep: c16 → c32 → c48 → c64 → c72 → c80 (warmup pass before each recorded stage)
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
