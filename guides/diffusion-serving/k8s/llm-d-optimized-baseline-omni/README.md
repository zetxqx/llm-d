# llm-d optimized-baseline for vLLM-Omni (diffusion)

An [optimized-baseline](../../../optimized-baseline/README.md) llm-d
deployment adapted to serve a **vLLM-Omni diffusion** model
(`Tongyi-MAI/Z-Image-Turbo`) behind the llm-d Router / EndpointPicker, instead of
the stock autoregressive LLM (`Qwen3-32B`).

It is the llm-d-routed counterpart of the standalone
[`vllm-omni-image.yaml`](../vllm-omni-image.yaml): same model and image,
but fronted by the llm-d inference scheduler with multiple replicas so we can
study how diffusion serving benefits (or doesn't) from llm-d routing.

## Structure

Same two-part shape as the upstream guide — a Helm-deployed **router** and a
Kustomize-deployed **model server** — kept as a thin diff over the
`guides/recipes` base.

```
router/omni.values.yaml                          # Z-Image router (openai-parser)
router/qwen-image.values.yaml                    # Qwen-Image router (openai-parser)
modelserver/gpu/vllm-omni/base/                  # Z-Image decode overlay (L4)
modelserver/gpu/vllm-omni/gke/                   # Z-Image GKE overlay (NCCL-tuner disable)
modelserver/gpu/vllm-omni-qwen-image/base/       # Qwen-Image decode overlay (H100)
modelserver/gpu/vllm-omni-qwen-image/gke/        # Qwen-Image GKE overlay
```

> The Kustomize overlays reference the repo recipes via relative paths
> (`../../../../../../../recipes/...`). They are validated with `kubectl kustomize`.

## What changed vs. stock optimized-baseline — and why it matters

This is the crux of the diffusion investigation:

| Aspect | Stock optimized-baseline (LLM) | This overlay (diffusion) |
| --- | --- | --- |
| Model | `Qwen/Qwen3-32B` | `Tongyi-MAI/Z-Image-Turbo` |
| GPUs / replica | 2 (TP=2) | 1 |
| Replicas | 8 | 2 (dev-friendly) |
| Router config | default LLM parser + queue/kv-cache/prefix-cache/no-hit-lru scorers | `payload-agnostic.yaml`: passthrough-parser + active-request-scorer + session-affinity |
| Image | `vllm/vllm-openai` | `vllm/vllm-omni:v0.22.0` (pinned in the overlay) |

Two independent reasons the stock LLM router can't route diffusion — **both
observed live on 2026-07-02** while bringing this up:

1. **Request parsing.** The stock EPP registers a request-body parser only for
   LLM path suffixes (`chat/completions`, `completions`). A diffusion request to
   `/v1/images/generations` matches no parser and the EPP rejects it with
   `HTTP 400: no parser registered matching path suffix for:
   /v1/images/generations`. The fix is the **passthrough parser** (no body
   parsing).

2. **Scoring.** The stock cache-aware scorers read vLLM *LLM* metrics the
   diffusion path never emits (no token KV cache, no prompt prefix cache).
   Even `queue-scorer` reads the backend `num_requests_waiting` metric, which
   the diffusion server does **not** export — so it has no signal. The
   load-aware scorer that works *without* backend metrics is
   **`active-request-scorer`**: the EPP counts in-flight requests it dispatched
   to each endpoint and prefers the least busy one.

The chart ships exactly this combination as **`payload-agnostic.yaml`**
(passthrough-parser + active-request-scorer + session-affinity-scorer), so
`router/omni.values.yaml` just points `epp.pluginsConfigFile` at it. This is the
honest subset of optimized-baseline that applies to diffusion today — the
baseline to measure before exploring diffusion-specific scorers (e.g. scoring by
denoising-steps-in-flight or VAE memory pressure).

## Image note

The `gpu-vllm-omni` image component defaults to a dev image
(`ghcr.io/revit13/vllm-openai:14d40dc7d`). This overlay overrides it to the
released `vllm/vllm-omni:v0.22.0` (matching the standalone `k8s/vllm-omni-image.yaml`)
via the `images:` block in `modelserver/gpu/vllm-omni/base/kustomization.yaml` —
no need to edit the cloned llm-d repo.

## Qwen-Image variant (H100)

`modelserver/gpu/vllm-omni-qwen-image/` is a second model-server overlay that
serves `Qwen/Qwen-Image` instead of `Z-Image-Turbo`. Qwen-Image is a ~20B MMDiT
plus a ~8B Qwen2.5-VL text encoder (~57 GB in fp16), so it does not fit the
24 GB L4. This overlay targets a single **80 GB H100** at full precision — no
quantization or CPU offload — via an H100 `nodeSelector`
(`cloud.google.com/gke-nodepool: bobbm-spoth100`; change to your pool). It pins
`vllm/vllm-omni:v0.24.0`, since Qwen-Image needs a newer image than the Z-Image
stack's v0.22.0.

It uses a distinct pool label (`llm-d.ai/guide: optimized-baseline-omni-qwen-image`)
and its own `router/qwen-image.values.yaml`, so it never shares endpoints with the
Z-Image pool. Both routers use the same custom EPP whose openai-parser claims
`/v1/images/generations`.

```bash
# Router for the Qwen-Image pool.
helm upgrade -i llm-d-omni-qwen-image $ROUTER_STANDALONE_CHART \
  -f $REPO_ROOT/guides/recipes/router/base.values.yaml \
  -f $GUIDE_DIR/router/qwen-image.values.yaml \
  -n $NAMESPACE --version $ROUTER_CHART_VERSION

# Model server (H100).
kubectl apply -n $NAMESPACE -k $GUIDE_DIR/modelserver/gpu/vllm-omni-qwen-image/gke/

# Generate through the router (Qwen-Image's recommended resolution is 1328x1328).
kubectl port-forward -n $NAMESPACE svc/llm-d-omni-qwen-image-epp 8080:80
curl -s -X POST http://localhost:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen-Image","prompt":"a pig over the Green Mountains of Vermont","size":"1328x1328","seed":42}' \
  | jq -r '.data[0].b64_json' | base64 -d > qwen-out.png
```

## Deploy

```bash
# from the repo root
export REPO_ROOT=$(pwd)
export NAMESPACE=llm-d-omni
export GUIDE_DIR=$REPO_ROOT/guides/diffusion-serving/k8s/llm-d-optimized-baseline-omni
source $REPO_ROOT/guides/env.sh

# 0. Prereqs: Gateway API Inference Extension CRDs + namespace (see the upstream
#    guide's Prerequisites). Z-Image-Turbo is public, so the HF token is
#    optional; create it only for gated models.
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 1. Router (standalone mode). Uses env vars from guides/env.sh.
helm upgrade -i llm-d-omni $ROUTER_STANDALONE_CHART \
  -f $REPO_ROOT/guides/recipes/router/base.values.yaml \
  -f $GUIDE_DIR/router/omni.values.yaml \
  -n $NAMESPACE --version $ROUTER_CHART_VERSION

# 2. Model server (GKE overlay; use base/ off-GKE).
kubectl apply -n $NAMESPACE -k $GUIDE_DIR/modelserver/gpu/vllm-omni/gke/
```

> After any change to `router/omni.values.yaml`, run `helm upgrade` (same flags
> as install) **and** `kubectl rollout restart deploy/llm-d-omni-epp -n $NAMESPACE`
> — the EPP loads its plugin config from a ConfigMap at startup and won't pick up
> changes without a restart.

## Verify

The `llm-d-omni-epp` service is ClusterIP, so port-forward it to localhost and
save the generated image on your machine.

```bash
# Terminal 1 — forward the router's HTTP port (80) to localhost:8080.
# Leave this running.
kubectl port-forward -n $NAMESPACE svc/llm-d-omni-epp 8080:80
```

```bash
# Terminal 2 — generate an image through the router and write it locally.
curl -s -X POST http://localhost:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Tongyi-MAI/Z-Image-Turbo","prompt":"a pig over the Green Mountains of Vermont","size":"256x256","seed":42}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png

# Inspect it locally:
file out.png    # -> PNG image data, 1024 x 1024
open out.png    # macOS  (Linux: xdg-open out.png)
```

### Via `/v1/chat/completions`

vLLM-Omni also serves diffusion models through the OpenAI chat API: the prompt
goes in as a user message and the image comes back as a base64 `data:` URL in
`choices[0].message.content`. Generation controls (`height`, `width`,
`num_inference_steps`, `seed`, ...) go inside an `extra_body` object rather than
top-level fields. Since the router uses the passthrough parser, this path routes
through the EPP the same way `/v1/images/generations` does.

```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Tongyi-MAI/Z-Image-Turbo",
    "messages": [
      {"role": "user", "content": "a pig over the red Mountains of Vermont"}
    ],
    "extra_body": {"height": 1024, "width": 1024, "seed": 42}
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > out.png
```

**Verified working 2026-07-02:** `HTTP 200`, ~3.1 MB PNG returned end-to-end
through the EPP router with 2 decode replicas Ready.

More generation examples (verified 2026-07-07, ~58s for 768x768 and ~28s for
512x512 on the L4 replicas):

```bash
# /v1/images/generations with a non-default size
curl -s -X POST http://localhost:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"Tongyi-MAI/Z-Image-Turbo","prompt":"a lighthouse on a rocky coast at sunset, oil painting style","size":"768x768","seed":7}' \
  | jq -r '.data[0].b64_json' | base64 -d > gen-lighthouse.png

# /v1/chat/completions with size controls in extra_body
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Tongyi-MAI/Z-Image-Turbo",
    "messages": [
      {"role": "user", "content": "a red fox sleeping in a snowy forest, watercolor style"}
    ],
    "extra_body": {"height": 512, "width": 512, "seed": 123}
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > chat-fox.png
```

### Via `/v1/images/edits` (img2img)

vLLM-Omni also serves the OpenAI image-edit API. For Z-Image-Turbo this is
**img2img**, not instruction editing: the input image is used as the starting
latent and `strength` (0-1) controls how far the result may drift from it
(higher = more change). One input image per request; the request is
`multipart/form-data`, not JSON, and again routes through the passthrough
parser like the other paths.

```bash
# Re-imagine an existing image under a new prompt.
curl -s -X POST http://localhost:8080/v1/images/edits \
  -F "image=@gen-lighthouse.png" \
  -F "prompt=the same lighthouse scene in winter, covered in snow under a night sky" \
  -F "model=Tongyi-MAI/Z-Image-Turbo" \
  -F "strength=0.7" \
  -F "size=512x512" \
  -F "seed=99" \
  | jq -r '.data[0].b64_json' | base64 -d > edit-lighthouse-winter.png
```

**Verified working 2026-07-07:** `HTTP 200` in ~41s; the output kept the
lighthouse composition and re-rendered it as a snow-covered winter scene.

### Response formats

Actual response bodies captured through the router on 2026-07-07, with the
base64 image payload omitted.

`/v1/images/generations` and `/v1/images/edits` share the same shape — the
image is in `data[0].b64_json`:

```json
{
  "created": 1783463245,
  "data": [
    {
      "b64_json": "<base64 PNG omitted>",
      "url": null,
      "revised_prompt": null
    }
  ],
  "output_format": "png",
  "size": "512x512",
  "cot_output": null
}
```

`/v1/chat/completions` returns a standard chat completion where
`choices[0].message.content` is a list with one `image_url` part; the image is
a base64 `data:` URL in `image_url.url`. vLLM-Omni adds per-part timing and
memory fields next to it:

```json
{
  "id": "chatcmpl-5583dc10c835423c",
  "object": "chat.completion",
  "created": 1783463245,
  "model": "Tongyi-MAI/Z-Image-Turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,<omitted>"
            },
            "stage_durations": {
              "queue_wait_ms": 0.30,
              "stage_0_gen_ms": 26478.06
            },
            "peak_memory_mb": 22244.0
          }
        ],
        "refusal": null,
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "total_tokens": 8,
    "completion_tokens": 1
  }
}
```

(The chat response also carries the usual vLLM null-valued fields —
`logprobs`, `system_fingerprint`, `kv_transfer_params`, `metrics`, ... —
trimmed here for readability.)

### Other endpoints

`GET /v1/models` also works through the router and lists
`Tongyi-MAI/Z-Image-Turbo`:

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```
