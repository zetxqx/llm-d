# GLM-5.2-FP8 on H200 with SGLang (Rust frontend, `/generate`)

This page is the engine reference for the SGLang variant (`MODEL_SERVER=sglang`) of the [GLM-5.2 guide](../../../README.md). Deploy, verify and clean up with that guide; this page explains how the recipe works, what its engine settings mean, how to tune it, and what does not work yet.

> [!NOTE]
> The recipe pins the SGLang nightly `nightly-dev-20261014-684fea8a`, because the features it needs are only in SGLang `main`. There is no benchmark data yet.

## How llm-d Talks to SGLang

llm-d treats every attention-DP rank as its own endpoint (pod IP x port) so it can route per rank. Two SGLang choices make that work without a Python frontend in the middle:

- **Rust frontend** (`SGLANG_RUST_SERVER=1`). Each DP rank embeds a Rust HTTP listener on `--port` + local rank, so a pod with 8 ranks listens on 8000 to 8007. Multi-node groups reuse the same port range on every pod ([sgl-project/sglang#34430](https://github.com/sgl-project/sglang/pull/34430)).
- **Native `/generate` API**. The Rust listener serves SGLang's `/generate` with pre-tokenized `input_ids`. The EPP parses these bodies with the `sglanghttp-parser` for prefix-cache scoring, and the routing sidecar disaggregates them ([llm-d/llm-d-router#2525](https://github.com/llm-d/llm-d-router/pull/2525)).

Request flow for one `/generate` call:

1. The EPP picks a prefill rank (`prefillIP:800r`) and a decode rank (`decodeIP:800r`) and sends the request to the decode pod's routing sidecar.
2. The sidecar (`--kv-connector=sglang`) adds `bootstrap_host` (the prefill pod IP), `bootstrap_port` (8000) and a random `bootstrap_room`, then sends the same body to the prefill rank and to its local decode rank (`localhost:820r`) at the same time.
3. The prefill rank registers the room it served with the bootstrap registry. In Rust mode the registry is served by the rank-0 listener (port 8000) of every prefill pod ([sgl-project/sglang#36234](https://github.com/sgl-project/sglang/pull/36234)). Each listener sends requests straight to its own rank, so `--load-balance-method` only selects how decode finds the prefill rank: the P/D default `follow_bootstrap_room` assumes `bootstrap_room % dp_size` and fails other ranks, while `round_robin` makes the rank register the room.
4. The decode rank looks the room up on `bootstrap_host:8000`, pulls the KV cache over NIXL, and streams tokens back through the sidecar.

## Engine Configuration

| Parameter | Value |
| --- | --- |
| Model | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) |
| Accelerator | NVIDIA H200 (8 GPUs per node) |
| Frontend | SGLang Rust server, one listener per DP rank (`SGLANG_RUST_SERVER=1`) |
| API | `/generate` with `input_ids` |
| Prefill parallelism | DEP8: attention DP=8, EP=8 (`--tp-size 8 --attn-dp-size 8 --ep-size 8`), 1 node |
| Decode parallelism | DEP16 (wide): attention DP=16, EP=16 (`--tp-size 16 --attn-dp-size 16 --ep-size 16`), 2 nodes |
| All-to-all | DeepEP v2 over NCCL GIN, hybrid mode, on both roles (`--moe-a2a-backend deepep_v2 --deepep-v2-mode hybrid`) |
| KV cache | FP8 (`--kv-cache-dtype fp8_e4m3`, `--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`) |
| KV transfer | NIXL over UCX (`DISAGG_TRANSFER_BACKEND`) |
| MTP speculative decoding | On (EAGLE, 3 steps, 4 draft tokens; draft MoE without all-to-all; opt-out via `no-mtp` component) |
| `mem-fraction-static` | 0.85 on both roles |
| Decode running requests | 32 per rank (`DECODE_MAX_RUNNING_REQUESTS_PER_RANK`, passed to SGLang as per-rank x DP), CUDA graph max batch 32 |
| Prefill chunk | 8192 tokens per rank (`PREFILL_CHUNK_TOKENS_PER_RANK`) |
| Reasoning / tool-call parsers | glm45 / glm47 (only used by OpenAI-style routes, see [Limitations](#limitations)) |

SGLang's `--tp-size` is the size of the GPU group that holds one copy of the model, not the attention TP size. Attention runs one DP rank per GPU inside that group (attention TP=1) and the experts are split across all its GPUs. The vLLM recipe spells the same layout `--tensor-parallel-size 1 --data-parallel-size N --enable-expert-parallel`.

SGLang defaults DSA models (GLM-5.2 uses DeepSeek Sparse Attention) to a bfloat16 KV cache on Hopper and only picks FP8 on Blackwell, so the recipe sets `--kv-cache-dtype fp8_e4m3` explicitly. That nearly doubles the KV capacity per GPU (656 instead of 1152 bytes per token); prefill and decode must use the same dtype. The recipe also pins both DSA backends to `flashmla_kv`, which is what SGLang picks for an FP8 KV cache on Hopper, so the kernels do not change with SGLang defaults. On Blackwell, SGLang would pick `trtllm` instead.

DeepEP v2 (`ElasticBuffer`) moves tokens with NCCL GIN (GPU-initiated networking) instead of NVSHMEM, so the recipe does not load NVSHMEM's IBGDA transport and runs the stock SGLang image on RoCE. Hybrid mode is required because decode spans nodes; the `direct` mode is NVLink only. SGLang does not allow DeepEP v2 for the MTP draft layer, so `MTP_ARGS` sets `--speculative-moe-a2a-backend none` and the one draft MoE layer runs without an all-to-all.

DeepEP v2 dispatches at most `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` tokens per rank per step, and SGLang checks the budget at startup. Decode uses 128: 32 running requests x 4 draft tokens; raise all three together. Prefill uses 8192, one full per-rank prefill chunk (`PREFILL_CHUNK_TOKENS_PER_RANK`); keep it at least that large.

## Layout

```text
prefill pod                                decode pod
+--------------------------------+         +----------------------------------------+
| sglang (Rust frontend)         |         | routing-proxy   8000 8001 ... 8007     |
|  rank0 8000  <- bootstrap      |         |   --kv-connector=sglang                |
|  rank1 8001     registry       |         |   SGLANG_BOOTSTRAP_PORT=8000           |
|  ...                           |         |        |    |          |               |
|  rank7 8007                    |         | sglang  8200 8201 ... 8207 (Rust)      |
+--------------------------------+         +----------------------------------------+
```

The InferencePool lists ports 8000 to 8007 (`router/glm-5-2.values.yaml`), so the EPP sees every rank of every pod as an endpoint.

## Tunables

All engine knobs are container env vars on the `sglang` container of each role, so overlays change them with name-based env merges as the `no-mtp` component does.

| Variable | Role | Default | Purpose |
| --- | --- | --- | --- |
| `DP_SIZE_LOCAL` | both | `8` | Ranks per pod; `--tp-size`, `--attn-dp-size` and `--ep-size` are all `DP_SIZE_LOCAL` x group size |
| `ENABLE_MTP`, `MTP_NUM_STEPS`, `MTP_NUM_DRAFT_TOKENS` | both | `1`, `3`, `4` | EAGLE/MTP speculative decoding |
| `PREFILL_MEM_FRACTION_STATIC`, `DECODE_MEM_FRACTION_STATIC` | prefill, decode | `0.85` | `--mem-fraction-static` |
| `PREFILL_CHUNK_TOKENS_PER_RANK` | prefill | `8192` | Per-rank chunked prefill budget; SGLang divides `--chunked-prefill-size` by DP, so the manifest passes per-rank x DP |
| `PREFILL_MAX_RUNNING_REQUESTS_PER_RANK` | prefill | `64` | Per-rank running requests; SGLang splits `--max-running-requests` across DP, so the manifest passes per-rank x DP |
| `DECODE_MAX_RUNNING_REQUESTS_PER_RANK`, `DECODE_CUDA_GRAPH_MAX_BS` | decode | `32`, `32` | Per-rank decode batch (passed as per-rank x DP); bounded by the DeepEP dispatch cap |
| `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | prefill, decode | `8192`, `128` | DeepEP v2 per-rank token buffer; prefill must hold one per-rank chunk |
| `CONTEXT_LENGTH` | both | unset | `--context-length`; unset keeps the model default |
| `DISAGG_TRANSFER_BACKEND` | both | `nixl` | `--disaggregation-transfer-backend` |

## Startup and Health

A pod is ready once `/health` on its local rank-0 listener returns `200`. Every Rust listener answers `503` until its own startup warmup, a one-token generation on its rank, succeeds ([sgl-project/sglang#43004](https://github.com/sgl-project/sglang/pull/43004)). The warmup uses SGLang's fake bootstrap host, so at startup the prefill log shows one `Fake KVSender Exception` line per rank; that is the warmup, not a failing request.

## Requests

`/generate` requests must carry `input_ids`. The EPP's `sglanghttp-parser` rejects bodies that only have `text`, because prefix-cache routing needs the token IDs. The guide's verification step tokenizes the prompt inside a prefill pod, which already has the tokenizer files in its HF cache. The response is SGLang's native format: generated `text` plus `meta_info` with `prompt_tokens`, `completion_tokens` and `cached_tokens`. Add `"stream": true` for server-sent events.

## Check the P/D Path

Every prefill rank listener must expose the full bootstrap topology. Query the rank-0 listener of each prefill pod (replace the pod IP):

```bash
curl -s "http://<prefill-pod-ip>:8000/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1" | jq
```

A `200` with one row per DP rank means the registry is populated. A `503` means the pod's listener has no topology; check that the image contains sgl-project/sglang#36234.

## Limitations

- **No per-rank `/metrics`.** The Rust listeners do not expose `/metrics`, so the EPP config in `router/sglang.values.yaml` uses only the approximate prefix-cache scorer and the active-request scorer. Queue depth and KV utilization scoring, and the guide's monitoring overlay, do not apply. Tracked in [llm-d/llm-d#2198](https://github.com/llm-d/llm-d/issues/2198).
- **`/generate` only, pre-tokenized.** The EPP parser list only contains `sglanghttp-parser`, so OpenAI-style paths are rejected with `400` and `/generate` bodies need `input_ids`. Chat and completions on the Rust frontend are follow-up work ([sgl-project/sglang#36718](https://github.com/sgl-project/sglang/pull/36718)); once the image serves them, add `openai-parser` to the EPP parser list.
- **No KV-event based precise routing.** Per-rank KV-cache ownership for shared endpoints is still being designed ([llm-d/llm-d-router#2306](https://github.com/llm-d/llm-d-router/issues/2306)), so there is no `kv-events` component here.
- **Do not switch back to DeepEP v1 on RoCE.** The SGLang image ships NVSHMEM 3.4.5 (pinned by PyTorch), whose IBGDA transport fails on RoCE with `create DCT share err` / `nvshmem setup connections failed` (uninitialized `ibv_ah_attr`; fixed upstream in NVSHMEM 3.5). `--moe-a2a-backend deepep` and `--speculative-moe-a2a-backend deepep` both use that transport, so multi-node decode cannot start on GKE A3 Ultra with them. DeepEP v2 does not use NVSHMEM.
- **Latency floor on the P/D path.** With the Rust frontend, a single `/generate` through P/D takes about 3s end to end even for a 5-token prompt, and the latency is quantized in 1s steps (3.0s, 4.0s, 5.0s). Long generations are fast once started (256 tokens in about 5.6s at 1 request), so the cost sits in the bootstrap and KV-transfer phase; the suspected cause is the Rust server's 1s idle tick gating the decode scheduler's bootstrap polling. Treat TTFT numbers from this recipe as not representative until this is resolved upstream.
- **DeepEP v2 is untuned.** `SGLANG_DEEPEP_V2_NUM_SMS` stays at its default and the draft MoE layer runs without an all-to-all. Throughput numbers need tuning before they are comparable with the vLLM recipe.
- **Nightly images.** The base pins the SGLang nightly by digest (`nightly-dev-20261014-684fea8a`) and tracks the `routing-sidecar/nightly` image component. Pin the sidecar in an overlay for anything beyond experimentation.
