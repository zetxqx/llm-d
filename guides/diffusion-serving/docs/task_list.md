# Project Task List: vLLM-Omni Support in EPP

This file contains a comprehensive task list for all integration phases of vLLM-Omni support into the llm-d inference scheduler.

## Phase 0 — API Enablement (Passthrough)
Goal: Ensure any vLLM-Omni request flows through the EPP without 400 errors.

- [ ] **Default fallback to passthrough parser**
  - Implement fallback handling for unhandled/unknown endpoints to route requests to a pod instead of returning `HTTP 400`.
- [ ] **Skip response processing for fallback**
  - Configure the passthrough path to set `SkipResponseProcessing = true` to protect binary and streaming outputs.

## Phase 1 — Single-Stage Diffusion Models
Goal: Support routing and cost estimation for single-stage diffusion models.

- [ ] **Implement parsers for single-stage diffusion endpoints**
  - [ ] Support `POST /v1/images/generations` (JSON): parse `model`, `size`, `num_inference_steps`, `n`.
  - [ ] Support `POST /v1/chat/completions` (JSON): parse `model`, `modalities`, `extra_body.num_inference_steps`, etc.
  - [ ] **(Stretch Goal)** Support `POST /v1/images/edits` (multipart form): parse `model`, `size`, `num_inference_steps`, `n`.
  - [ ] **(Stretch Goal)** Support `POST /v1/videos` & `POST /v1/videos/sync` (multipart form): parse `model`, `size`, `seconds`, `fps`, `num_inference_steps`.
- [ ] **Wire metrics through EPP data layer**
  - Add `vllm-omni` engine configuration.
  - Map `vllm_omni:num_requests_waiting` -> `WaitingQueueSize`.
  - Map `vllm_omni:num_requests_running` -> `RunningRequestsSize`.
  - Support empty KV cache specs for diffusion-only pools.
- [ ] **Implement cost-aware scheduling scorer**
  - Implement `diffusioncost` scorer plugin.
  - Route based on backlog in GPU-seconds (sum of declared costs of queued + running requests) using formula: `cost ≈ num_inference_steps × f(resolution) × n`.
- [ ] **Add diffusion benchmark tools and SLO measurement**
  - Integrate a load generator to run Poisson-distributed heterogeneous requests and measure p99 and SLO attainment rate.

## Phase 2 — TTS and Omni Serving (Aggregated)
Goal: Support text-to-speech (TTS) routing and investigate prefix-cache benefits.

- [ ] **Implement parser for audio endpoints**
  - Support `POST /v1/audio/speech` (JSON): extract `model`, `input`, `voice`, and `ref_audio` / `ref_text`.
- [ ] **Benchmark and evaluate default EPP configurations on TTS**
  - Run a TTS benchmark comparing EPP default config (using `vllm:*` metrics and prefix caching) against random routing to verify scorer efficacy.
- [ ] **(Stretch Goal) Support Realtime/WebSocket connections**
  - [ ] Enable EPP to detect `Connection: upgrade` headers and route instantly without waiting for a request body.
  - [ ] Implement session affinity (pinning) for WebSocket connections.

## Phase 3 — Disaggregated Serving (One Pool per Stage)
Goal: Package and route requests across multi-stage vLLM-Omni deployments.

- [ ] **Package each stage as its own pool**
  - Create Kubernetes manifests for stage vllmomni-worker deployments.
- [ ] **Implement NIXL backend for OmniConnector**
  - Align vLLM-Omni's inter-stage tensor transfers with llm-d's NIXL connector.
- [ ] **Implement cluster routing layer**
  - Sandalone coordination service with one EPP per stage pool.
