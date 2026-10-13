# GLM-5.2-FP8 on H200 with SGLang (Rust frontend, `/generate`)

## Overview

This recipe deploys [GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) (753B MoE) on H200 GPUs with SGLang, P/D disaggregated, in a wide expert parallel pattern with DP-aware scheduling. Prefill runs DEP8 (attention DP=8, EP=8) on 1 node by default; decode runs DEP16 (attention DP=16, EP=16) across 2 nodes. DeepEP `normal` mode for prefill and `low_latency` mode for decode.

It is the SGLang variant (`MODEL_SERVER=sglang`) of the [GLM-5.2 guide](../../../README.md) and shares its router values and deploy steps with the vLLM recipe. It does not aim for feature parity with the vLLM recipe. Tracking issues: [llm-d/llm-d#2198](https://github.com/llm-d/llm-d/issues/2198) and [llm-d/llm-d#2554](https://github.com/llm-d/llm-d/issues/2554).

> [!WARNING]
> This is an initial recipe. It was brought up once end to end on GKE A3 Ultra (3 nodes, `p1w1d1w2`, Spot H200) with the NVSHMEM-fixed image described in [Limitations](#limitations); no benchmark numbers are published yet. The SGLang pieces it depends on are only in SGLang `main` (nightly image), not in a release.

### How llm-d talks to SGLang here

llm-d treats every attention-DP rank as its own endpoint (pod IP x port) so it can route per rank. Two SGLang choices make that work without a Python frontend in the middle:

- **Rust frontend** (`SGLANG_RUST_SERVER=1`). Each DP rank embeds a Rust HTTP listener on `--port` + local rank, so a pod with 8 ranks listens on 8000 to 8007. Multi-node groups reuse the same port range on every pod ([sgl-project/sglang#34430](https://github.com/sgl-project/sglang/pull/34430)).
- **Native `/generate` API**. The Rust listener serves SGLang's `/generate` with pre-tokenized `input_ids`. The EPP parses these bodies with the `sglanghttp-parser` for prefix-cache scoring, and the routing sidecar disaggregates them ([llm-d/llm-d-router#2525](https://github.com/llm-d/llm-d-router/pull/2525)).

Request flow for one `/generate` call:

1. The EPP picks a prefill rank (`prefillIP:800r`) and a decode rank (`decodeIP:800r`) and sends the request to the decode pod's routing sidecar.
2. The sidecar (`--kv-connector=sglang`) adds `bootstrap_host` (the prefill pod IP), `bootstrap_port` (8000) and a random `bootstrap_room`, then sends the same body to the prefill rank and to its local decode rank (`localhost:820r`) at the same time.
3. The prefill rank registers the room it served with the bootstrap registry. In Rust mode the registry is served by the rank-0 listener (port 8000) of every prefill pod ([sgl-project/sglang#36234](https://github.com/sgl-project/sglang/pull/36234)). Each listener sends requests straight to its own rank, so `--load-balance-method` only selects how decode finds the prefill rank: the P/D default `follow_bootstrap_room` assumes `bootstrap_room % dp_size` and fails other ranks, while `round_robin` makes the rank register the room.
4. The decode rank looks the room up on `bootstrap_host:8000`, pulls the KV cache over NIXL, and streams tokens back through the sidecar.

## Default Configuration

| Parameter | Value |
| --- | --- |
| Model | [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) |
| Accelerator | NVIDIA H200 (8 GPUs per node) |
| Frontend | SGLang Rust server, one listener per DP rank (`SGLANG_RUST_SERVER=1`) |
| API | `/generate` with `input_ids` |
| Prefill parallelism | DEP8: attention DP=8, EP=8 (`--tp-size 8 --attn-dp-size 8 --ep-size 8`), 1 node |
| Decode parallelism | DEP16 (wide): attention DP=16, EP=16 (`--tp-size 16 --attn-dp-size 16 --ep-size 16`), 2 nodes |
| All-to-all | DeepEP, `normal` (prefill) and `low_latency` (decode) |
| KV cache | FP8 (`--kv-cache-dtype fp8_e4m3`, `--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`) |
| KV transfer | NIXL (`DISAGG_TRANSFER_BACKEND`; only `nixl` has been tested) |
| MTP speculative decoding | On (EAGLE, 3 steps, 4 draft tokens; opt-out via `no-mtp` component) |
| `mem-fraction-static` | 0.85 on both roles |
| Decode running requests | 32 per rank (`DECODE_MAX_RUNNING_REQUESTS_PER_RANK`, passed to SGLang as per-rank x DP), CUDA graph max batch 32 |
| Prefill chunk | 8192 tokens per rank (`PREFILL_CHUNK_TOKENS_PER_RANK`) |
| Reasoning / tool-call parsers | glm45 / glm47 (only used by OpenAI-style routes, see [Limitations](#limitations)) |

SGLang's `--tp-size` is the size of the GPU group that holds one copy of the model, not the attention TP size. Attention runs one DP rank per GPU inside that group (attention TP=1) and the experts are split across all its GPUs. The vLLM recipe spells the same layout `--tensor-parallel-size 1 --data-parallel-size N --enable-expert-parallel`.

SGLang defaults DSA models (GLM-5.2 uses DeepSeek Sparse Attention) to a bfloat16 KV cache on Hopper and only picks FP8 on Blackwell, so the recipe sets `--kv-cache-dtype fp8_e4m3` explicitly. That nearly doubles the KV capacity per GPU (656 instead of 1152 bytes per token); prefill and decode must use the same dtype. The recipe also pins both DSA backends to `flashmla_kv`, which is what SGLang picks for an FP8 KV cache on Hopper, so the kernels do not change with SGLang defaults. On Blackwell, SGLang would pick `trtllm` instead.

The decode limits follow from DeepEP low-latency mode, which dispatches at most `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (128) tokens per rank per step: 32 running requests x 4 draft tokens. Raise all three together.

### P/D Deployment Options

Deployments are named `p<replicas>w<nodes>d<replicas>w<nodes>`; `default` is `p1w1d1w2`. All of them build on the GKE provider overlay, the only one so far; other providers need their own `providers/<name>` overlay (RDMA device requests, HCA selection, cache volumes).

| Deployment | Prefill | Decode | Nodes / GPUs |
| --- | --- | --- | --- |
| `p1w1d1w1` | 1 replica, 1 node, DEP8 | 1 replica, 1 node, DEP8 | 2 / 16 |
| `p1w1d1w2` | 1 replica, 1 node, DEP8 | 1 replica, 2 nodes, DEP16 | 3 / 24 |
| `p1w2d1w2` | 1 replica, 2 nodes, DEP16 | 1 replica, 2 nodes, DEP16 | 4 / 32 |
| `p2w1d1w2` | 2 replicas, 1 node, DEP8 | 1 replica, 2 nodes, DEP16 | 4 / 32 |
| `p2w2d2w2` | 2 replicas, 2 nodes, DEP16 | 2 replicas, 2 nodes, DEP16 | 8 / 64 |

### Supported Hardware Backends

| Backend | Directory | Notes |
| --- | --- | --- |
| NVIDIA GPU (GKE A3 Ultra, RoCE) | `providers/gke/`, `deployments/*` | GPUs and RDMA NICs through DRA (`gke-rdma` claims), privileged containers for GPU-initiated RDMA, block/subblock affinity; needs the NVSHMEM fix in [Limitations](#limitations) |

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

## Components

Add [kustomize Components](https://kubectl.docs.kubernetes.io/guides/config_management/components/) to a deployment's `kustomization.yaml` under `components:`.

| Component | Targets | Effect |
| --- | --- | --- |
| `no-mtp` | prefill + decode | Disables MTP speculative decoding (`ENABLE_MTP=0`) |

Component env entries merge by name, so their values replace the base defaults.

## Deploy

Follow the [GLM-5.2 guide](../../../README.md) with `MODEL_SERVER=sglang`. Its prerequisites, router install, model server step, verification and cleanup all have SGLang variants: the router step layers [`router/sglang.values.yaml`](../../../router/sglang.values.yaml) on top of `glm-5-2.values.yaml`, and `DEPLOYMENT` picks an overlay from [P/D Deployment Options](#pd-deployment-options) (`default` is `p1w1d1w2`). Only approximate prefix routing (`ROUTING=approximate`) applies.

For GKE, prepare the cluster as described in [the GKE provider docs](../../../../../../docs/infrastructure/providers/gke/README.md) (GPUDirect RDMA, DRA and DRANET drivers) so the `gke-rdma-template` claims resolve.

On GKE the stock nightly image cannot start the multi-node decode group (see [Limitations](#limitations)), so wrap the deployment in a small overlay that swaps in your NVSHMEM-fixed build, and apply that overlay instead of `deployments/${DEPLOYMENT}`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - <path-to>/sglang/deployments/p1w1d1w2
images:
  - name: docker.io/lmsysorg/sglang
    newName: <your-registry>/sglang-roce
    newTag: <tag>
```

Do not deploy this recipe and the vLLM recipe in the same namespace: the InferencePool selects pods by `llm-d.ai/guide` only and would mix both engines into one pool.

Model load takes time; the startup probe allows up to 45 minutes. A pod is ready once `/health` on its local rank-0 listener returns `200`. Every Rust listener answers `503` until its own startup warmup, a one-token generation on its rank, succeeds ([sgl-project/sglang#43004](https://github.com/sgl-project/sglang/pull/43004)). The warmup uses SGLang's fake bootstrap host, so at startup the prefill log shows one `Fake KVSender Exception` line per rank; that is the warmup, not a failing request.

### Requests

`/generate` requests must carry `input_ids`. The EPP's `sglanghttp-parser` rejects bodies that only have `text`, because prefix-cache routing needs the token IDs. The guide's verification step tokenizes the prompt inside a prefill pod, which already has the tokenizer files in its HF cache. The response is SGLang's native format: generated `text` plus `meta_info` with `prompt_tokens`, `completion_tokens` and `cached_tokens`. Add `"stream": true` for server-sent events.

### Check the P/D path

Every prefill rank listener must expose the full bootstrap topology. Query the rank-0 listener of each prefill pod (replace the pod IP):

```bash
curl -s "http://<prefill-pod-ip>:8000/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1" | jq
```

A `200` with one row per DP rank means the registry is populated. A `503` means the pod's listener has no topology; check that the image contains sgl-project/sglang#36234.

## Optional Features

### MTP Speculative Decoding

On by default for both roles as EAGLE with 3 steps and 4 draft tokens (GLM-5.2 ships one MTP layer). Disable with the `no-mtp` component or `ENABLE_MTP=0`; tune with `MTP_NUM_STEPS` and `MTP_NUM_DRAFT_TOKENS`.

### EPP Routing

`router/sglang.values.yaml` replaces the vLLM plugin config of `router/glm-5-2.values.yaml`. It parses `/generate` with `sglanghttp-parser`, scores prefill endpoints with the approximate prefix-cache scorer (weight 3) and the active-request scorer (weight 2), and scores decode endpoints with the active-request scorer. All 8 DP rank ports (8000-8007) are inherited as `targetPorts`. The metrics refresh interval is raised to 10s because the Rust listeners expose no `/metrics`.

### Tunables

All engine knobs are container env vars on the `sglang` container of each role, so overlays change them with name-based env merges as the `no-mtp` component does.

| Variable | Role | Default | Purpose |
| --- | --- | --- | --- |
| `DP_SIZE_LOCAL` | both | `8` | Ranks per pod; `--tp-size`, `--attn-dp-size` and `--ep-size` are all `DP_SIZE_LOCAL` x group size |
| `ENABLE_MTP`, `MTP_NUM_STEPS`, `MTP_NUM_DRAFT_TOKENS` | both | `1`, `3`, `4` | EAGLE/MTP speculative decoding |
| `PREFILL_MEM_FRACTION_STATIC`, `DECODE_MEM_FRACTION_STATIC` | prefill, decode | `0.85` | `--mem-fraction-static` |
| `PREFILL_CHUNK_TOKENS_PER_RANK` | prefill | `8192` | Per-rank chunked prefill budget; SGLang divides `--chunked-prefill-size` by DP, so the manifest passes per-rank x DP |
| `PREFILL_MAX_RUNNING_REQUESTS_PER_RANK` | prefill | `64` | Per-rank running requests; SGLang splits `--max-running-requests` across DP, so the manifest passes per-rank x DP |
| `DECODE_MAX_RUNNING_REQUESTS_PER_RANK`, `DECODE_CUDA_GRAPH_MAX_BS` | decode | `32`, `32` | Per-rank decode batch (passed as per-rank x DP); bounded by the DeepEP dispatch cap |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | decode | `128` | DeepEP low-latency per-rank token buffer |
| `CONTEXT_LENGTH` | both | unset | `--context-length`; unset keeps the model default |
| `DISAGG_TRANSFER_BACKEND` | both | `nixl` | `--disaggregation-transfer-backend` |

## Limitations

- **No per-rank `/metrics`.** The Rust listeners do not expose `/metrics`, so the EPP config in `router/sglang.values.yaml` uses only the approximate prefix-cache scorer and the active-request scorer. Queue depth and KV utilization scoring, and the guide's monitoring overlay, do not apply. Tracked in [llm-d/llm-d#2198](https://github.com/llm-d/llm-d/issues/2198).
- **`/generate` only, pre-tokenized.** The EPP parser list only contains `sglanghttp-parser`, so OpenAI-style paths are rejected with `400` and `/generate` bodies need `input_ids`. Chat and completions on the Rust frontend are follow-up work ([sgl-project/sglang#36718](https://github.com/sgl-project/sglang/pull/36718)); once the image serves them, add `openai-parser` to the EPP parser list.
- **No KV-event based precise routing.** Per-rank KV-cache ownership for shared endpoints is still being designed ([llm-d/llm-d-router#2306](https://github.com/llm-d/llm-d-router/issues/2306)), so there is no `kv-events` component here.
- **NVSHMEM on RoCE (GKE).** The SGLang nightly ships NVSHMEM 3.4.5, whose IBGDA transport fails on RoCE with `create DCT share err` / `nvshmem setup connections failed` (uninitialized `ibv_ah_attr`; fixed upstream in NVSHMEM 3.5). DeepEP low-latency mode on multi-node decode therefore cannot start on GKE A3 Ultra with the stock image. InfiniBand fabrics are not affected.
- **Fixed image for RoCE.** The recipe carries the one-line fix as `image/nvshmem_zero_ibv_ah_attr_v3.4.5-0.patch`. `image/Dockerfile` rebuilds only the `nvshmem_transport_ibgda.so.3` plugin from patched 3.4.5 source and copies it over the wheel's copy, so the DeepEP kernels keep their ABI. Build it and point the `docker.io/lmsysorg/sglang` image at your build with a kustomize `images:` override:

  ```bash
  # BASE_IMAGE must be the SGLang image the gpu-sglang/nightly component resolves to (pin it by digest).
  docker build --build-arg BASE_IMAGE=docker.io/lmsysorg/sglang@sha256:<digest> -t <your-registry>/sglang-roce:<tag> image/
  ```

- **Latency floor on the P/D path.** With the Rust frontend, a single `/generate` through P/D takes about 3s end to end even for a 5-token prompt, and the latency is quantized in 1s steps (3.0s, 4.0s, 5.0s). Long generations are fast once started (256 tokens in about 5.6s at 1 request), so the cost sits in the bootstrap and KV-transfer phase; the suspected cause is the Rust server's 1s idle tick gating the decode scheduler's bootstrap polling. Treat TTFT numbers from this recipe as not representative until this is resolved upstream.
- **DeepEP is untuned.** The engine logs `Only use 20 SMs for DeepEP communication`; no `--deepep-config` is set. Throughput numbers need tuning before they are comparable with the vLLM recipe.
- **Nightly images.** The base tracks the `gpu-sglang/nightly` and `routing-sidecar/nightly` image components. Pin them in an overlay for anything beyond experimentation.

## Cleanup

Use the cleanup steps of the [GLM-5.2 guide](../../../README.md#cleanup); if you applied your own image overlay, delete that overlay instead of `deployments/${DEPLOYMENT}`.
