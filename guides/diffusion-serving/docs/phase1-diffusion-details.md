# Phase 1 details — single-stage diffusion support

Supporting detail for Phase 1 of the plan (`docs/myplan.md`). Phase 1 has four
work items — parsers, metrics, cost-aware scorer, benchmark — and this doc has
one section per item, plus a section on which models are in scope.

All GitHub links are pinned to vllm-omni commit
[`1b318d1`](https://github.com/vllm-project/vllm-omni/tree/1b318d11d17804c54c6ffa482efdd7abcb03657c)
(2026-06-28, the version cloned in this workspace) so line numbers stay valid.

## 1. Which models this covers

Phase 1 targets models that run entirely inside vLLM-Omni's `DiffusionEngine`:
one process, one stage, no pipeline. Native implementations live under
[`vllm_omni/diffusion/models/`](https://github.com/vllm-project/vllm-omni/tree/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/diffusion/models);
the model → pipeline mapping is in
[`vllm_omni/diffusion/registry.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/diffusion/registry.py).

- **Text-to-image:** Qwen-Image, Z-Image, FLUX (`flux`, `flux2`,
  `flux2_klein`), SD3, SDXL, HiDream, OmniGen2, Ovis-Image, LongCat-Image,
  ERNIE-Image, NextStep-1.1.
- **Image editing:** Qwen-Image-Edit
  ([`pipeline_qwen_image_edit.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py)).
- **Text-to-video:** Wan2.2, HunyuanVideo, LTX2.
- **Audio diffusion:** Stable Audio, AudioX.
- **Any HuggingFace pipeline:** the
  [Diffusers adapter](https://github.com/vllm-project/vllm-omni/tree/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/diffusion/models/diffusers_adapter)
  loads unmodified HF Diffusers pipelines via
  `--diffusion-load-format diffusers`.

How to tell single-stage from multi-stage: single-stage diffusion models are
registered in `diffusion/registry.py` and run entirely inside
`DiffusionEngine`; multi-stage models (GLM-Image, BAGEL, HunyuanImage-3,
Ming-Flash-Omni) have a stage-graph yaml in
[`vllm_omni/deploy/`](https://github.com/vllm-project/vllm-omni/tree/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/deploy)
(BAGEL also ships a
[`bagel_single_stage.yaml`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/deploy/bagel_single_stage.yaml)
option).

Docs: [supported models list](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/).

## 2. Work item 1 — parsers (endpoints and request shapes)

All routes are defined in
[`vllm_omni/entrypoints/openai/api_server.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/openai/api_server.py);
line links below point into that file at the pinned commit. Per endpoint, the
fields the EPP parser should read:

- [**`POST /v1/images/generations`**](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/openai/api_server.py#L1709)
  (L1709, JSON body) — `model`, `prompt`, `n`, `size`,
  `num_inference_steps`, `seed`.
- [**`POST /v1/images/edits`**](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/openai/api_server.py#L1937)
  (L1937, multipart form) — same fields as generations, plus the input
  image.
- [**`POST /v1/videos`**](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/openai/api_server.py#L3046)
  (L3046, multipart form) — `model` (optional form field), `prompt`,
  `seconds`, `size`, `fps`, `num_inference_steps`, `guidance_scale`
  (`_parse_video_form`, L2894). This is an **async job API**:
  submit → poll → fetch.
- [**`POST /v1/videos/sync`**](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/openai/api_server.py#L3089)
  (L3089) — synchronous variant, same form fields.
- **`GET /v1/videos/{id}`** and **`GET /v1/videos/{id}/content`** — job poll
  and result fetch. Nothing to parse, but these must land on a pod that can
  see the job, so either sticky routing or shared job state is required.
- [**`POST /v1/chat/completions`**](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/entrypoints/openai/api_server.py#L1159)
  (L1159, JSON body) — diffusion requests also flow through chat: the
  parser should read `modalities` plus `extra_body.num_inference_steps`,
  `extra_body.guidance_scale`, and `extra_body.seed`. Image editing can go
  through this route too.

Notes for the parser design:

- Image edit responses return the image as a base64 data URL inside the chat
  message content; generation responses use the OpenAI Images shape
  (`data[0].b64_json`). Response parsing is out of scope (passthrough +
  `SkipResponseProcessing`).
- Runnable request examples per modality:
  [`examples/online_serving/`](https://github.com/vllm-project/vllm-omni/tree/1b318d11d17804c54c6ffa482efdd7abcb03657c/examples/online_serving).
- A first parser prototype already exists in the local
  llm-d-inference-scheduler fork: commit `2062b2b1` adds
  `/v1/images/generations` support to the openai-parser.

## 3. Work item 2 — metrics

vLLM-Omni exports its own Prometheus namespace, `vllm_omni:*`, defined in
[`vllm_omni/metrics/definitions.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/metrics/definitions.py)
(`METRIC_PREFIX = "vllm_omni:"` at L18) and exported by
[`vllm_omni/metrics/prometheus.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/metrics/prometheus.py).
The ones that matter for routing:

- `vllm_omni:num_requests_running` (gauge) — in-flight requests across
  pipeline stages.
- `vllm_omni:num_requests_waiting` (gauge) — queued requests.
- `vllm_omni:requests_success` (counter) — completions, labeled by finish
  reason (stop / length / abort).
- `vllm_omni:e2e_request_latency_s` (histogram) — arrival to complete
  response. Note the name ends in `_s`, not `_seconds`.
- `vllm_omni:prompt_tokens` and `vllm_omni:generation_tokens` (counters) —
  tokens per stage.

(The namespace also has audio-streaming and transfer metric families; those
matter for later phases, not Phase 1.)

### Wiring through the EPP data layer

The EPP's
[`core-metrics-extractor`](https://github.com/llm-d/llm-d-inference-scheduler/tree/main/pkg/epp/framework/plugins/datalayer/extractor/metrics)
data-layer plugin maps engine-specific metric names to standard endpoint
attributes (`WaitingQueueSize`, `RunningRequestsSize`, `KVCacheUsagePercent`,
…) that scorers consume. Built-in engine mappings live in
`factories.go` (`defaultEngineConfigs`): `vllm`, `sglang`, `trtllm-serve`,
`triton-tensorrt-llm`, `triton`. The engine is selected per pod via the
`llm-d.ai/engine-type` label.

vLLM-Omni support is the same shape as the SGLang entry. Two paths:

1. **Config-only (works today, no code change)** — declare the engine in the
   EPP config and label the pods:

   ```yaml
   type: core-metrics-extractor
   parameters:
     engineConfigs:
       - name: "vllm-omni"
         queuedRequestsSpec: "vllm_omni:num_requests_waiting"
         runningRequestsSpec: "vllm_omni:num_requests_running"
         # no KV / LoRA / cache specs for diffusion pools
   ```

   ```yaml
   metadata:
     labels:
       llm-d.ai/engine-type: vllm-omni
   ```

   Empty KV specs are supported — the built-in `triton` entry ships the same
   way.

2. **Built-in (small upstream PR)** — add a `vllm-omni` entry to
   `defaultEngineConfigs`, exactly parallel to the `sglang` one.

Notes:

- **AR-stage pools can mix namespaces.** Spec strings are just metric names,
  and an omni pod serves both families from one `/metrics` endpoint — so an
  AR-pool engine config can take queue depth from `vllm_omni:*` while keeping
  `kvUsageSpec: "vllm:kv_cache_usage_perc"` etc. from the underlying vLLM
  engine.
- **`customMetrics`** entries map any scalar metric to a named endpoint
  attribute — the natural hookup for the cost-aware scorer once a
  backlog-in-GPU-seconds gauge exists (that gauge is still the missing
  upstream piece on the vllm-omni side).

## 4. Work item 3 — cost-aware scorer

The scorer treats declared cost the way LLM routing treats in-flight tokens:
each pod's load is the sum of the declared costs of its queued + running
requests (backlog in GPU-seconds), and the request goes to the pod with the
least backlog — not the fewest requests.

```
cost ≈ num_inference_steps × f(resolution) × n
```

Why declared cost should work well here:

- The denoise loop tracks its position as `denoise_step_idx` in
  [`vllm_omni/diffusion/forward_context.py#L30`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/diffusion/forward_context.py#L30)
  — per-step latency is measurable and appears stable per (model,
  resolution).
- In the default execution mode, the diffusion engine runs **one request at
  a time**:
  [`vllm_omni/diffusion/diffusion_engine.py#L141-L143`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/vllm_omni/diffusion/diffusion_engine.py#L141-L143)
  clamps `max_num_running_reqs` to 1 whenever step-level execution is off.
  Backlog is then a strict FIFO queue, and `queue position × declared cost`
  should be a close wait-time estimate. Caveat: with step-level execution
  enabled (required for streaming output), the engine can interleave more
  than one request, and the estimate gets looser.

Second caveat: step caches (TeaCache / Cache-DiT, the `-O3` presets) skip
steps based on content — reportedly ~1.4–2.4× speedups — so declared cost is
an **upper bound**, not exact. It should still order pods correctly; a
refined estimator can learn each pool's average speedup factor.

Prototype that already exists in the local llm-d-inference-scheduler fork:
`pkg/epp/framework/plugins/scheduling/scorer/diffusioncost/` (commit
`6c936e3d`, "declared-cost load tracking and scoring for diffusion
requests") — starting point for upstreaming.

## 5. Work item 4 — benchmark

Load generator:
[`benchmarks/diffusion/diffusion_benchmark_serving.py`](https://github.com/vllm-project/vllm-omni/blob/1b318d11d17804c54c6ffa482efdd7abcb03657c/benchmarks/diffusion/diffusion_benchmark_serving.py)
— Poisson `--request-rate`, `--max-concurrency`, heterogeneous request
profiles, and built-in `--slo` / `slo_attainment_rate` with a linear
resolution×steps cost model. Run it in-cluster (it health-checks
`{base_url}/health`). Expected result shape: load-aware routing improves p99
and SLO attainment, not peak throughput.
