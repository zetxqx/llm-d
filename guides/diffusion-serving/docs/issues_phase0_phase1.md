# GitHub Issue: vLLM-Omni Support (Phase 0 & Phase 1)

This file contains the title and body for a single tracking GitHub issue containing a checklist of all the Phase 0 and Phase 1 tasks.

---

**Title:**
`[Tracking] vLLM-Omni Support: Phase 0 & Phase 1`

**Body:**
```markdown
This issue tracks the implementation of Phase 0 (API enablement/passthrough) and Phase 1 (Single-stage diffusion models) for supporting vLLM-Omni as a backend in the llm-d inference scheduler.

### Phase 0 — API Enablement (Passthrough)
Goal: Ensure any vLLM-Omni request flows through the EPP without 400 errors.

- [ ] **Fallback to default passthrough parser for unknown endpoints**
  - Make the passthrough parser the default fallback for unhandled paths (e.g. `/v1/images/generations` should not return HTTP 400).
- [ ] **Skip response processing for default passthrough parser**
  - Ensure the fallback passthrough parser sets `SkipResponseProcessing` to true so EPP doesn't touch binary (PNG, WAV, MP4) or streaming responses.

### Phase 1 — Single-Stage Diffusion Models
Goal: Support text-to-image, text-to-video, and other single-stage diffusion models.

- [ ] **Implement parsers for single-stage diffusion model endpoints**
  - Extract `model` and cost fields (e.g. steps, resolution, count) from:
    - `POST /v1/images/generations` (JSON)
    - `POST /v1/chat/completions` (via `extra_body` and `modalities`)
  - **Stretch Goals (parsers for multipart inputs and video):**
    - [ ] `POST /v1/images/edits` (multipart form)
    - [ ] `POST /v1/videos` & `POST /v1/videos/sync` (multipart form)
- [ ] **Wire vLLM-Omni metrics through the EPP data layer**
  - Add `vllm-omni` engine configuration support to the `core-metrics-extractor` plugin.
  - Map `vllm_omni:num_requests_waiting` -> `WaitingQueueSize`
  - Map `vllm_omni:num_requests_running` -> `RunningRequestsSize`
- [ ] **Implement cost-aware scheduling scorer for diffusion workloads**
  - Implement the `diffusioncost` scorer plugin.
  - Route requests based on backlog in GPU-seconds (sum of declared costs of queued + running requests) using the formula: `cost ≈ num_inference_steps × f(resolution) × n`.
- [ ] **Add diffusion benchmark tools and SLO measurement**
  - Run Poisson-distributed heterogeneous request load using `benchmarks/diffusion/diffusion_benchmark_serving.py`.
  - Measure p99 latency and SLO attainment rate to validate routing gains.

---
For more details, see:
- [Plan: llm-d support for vLLM-Omni](docs/myplan.md)
- [Phase 1 details — single-stage diffusion support](docs/phase1-diffusion-details.md)
```
