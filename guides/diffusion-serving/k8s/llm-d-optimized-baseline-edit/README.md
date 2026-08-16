# llm-d optimized-baseline for vLLM-Omni image editing (OmniGen2)

An llm-d stack serving **OmniGen2** (`OmniGen2/OmniGen2`) — instruction-based
**image editing** (image + text → edited image) via `POST /v1/chat/completions`
— behind the llm-d Router / EndpointPicker. Third stack in the investigation
fleet, alongside
[diffusion](../llm-d-optimized-baseline-omni/README.md) and
[TTS](../llm-d-optimized-baseline-tts/README.md).

## The three-stack routing matrix

Each stack fails / behaves differently at the router — that contrast is the
point of the fleet:

| | Diffusion (Z-Image) | TTS (Qwen3-TTS) | **Edit (OmniGen2)** |
| --- | --- | --- | --- |
| API path | `/v1/images/generations` | `/v1/audio/speech` | **`/v1/chat/completions`** |
| Stock EPP parser | ✗ 400 (no parser) | ✗ 400 (no parser) | **✓ parses** (openai path) |
| Backend cache signal | none (pure DiT) | **yes** (AR talker, prefix caching on) | none (pure DiT) |
| Why payload-agnostic | parser + scorers | parser (scorers TBD) | **body cost**: multi-MB base64 images; parsing/prefix-hashing pixels is waste |
| Session affinity value | low | medium | **high** — iterative editing returns to warm endpoint |

## Model choice

The repo's reference edit model (`Qwen/Qwen-Image-Edit`) is ~20B DiT + 7B text
encoder — far beyond a 24 GB L4. **OmniGen2** (~4B DiT + Qwen2.5-VL-3B) is the
editing-capable model that fits an L4 — but only with offload:

> **L4 OOM finding (2026-07-02):** without offload, the resident MLLM text
> encoder + DiT peak at ~22.0 GiB during the startup dummy run and OOM on the
> L4's 22.03 GiB (the repo table's ~14.7 GB figure doesn't cover this peak).
> The overlay therefore sets **`--enable-cpu-offload`** (idle components park
> in host RAM; some per-request latency cost) alongside
> `--vae-use-slicing --vae-use-tiling`. Drop the offload flag on ≥40 GB GPUs.
> This is also a live illustration of the heterogeneous-fleet argument in
> vLLM-Omni RFC #4590: the encoder doesn't need to live on the DiT's GPU.

Editing knobs (from
[`examples/offline_inference/image_to_image/image_edit.py`](../../vllm-omni/examples/offline_inference/image_to_image/image_edit.py)):
`guidance_scale` = **text** guidance, `guidance_scale_2` = **image** guidance.

## Structure

```
router/edit.values.yaml                         # EPP config (payload-agnostic)
modelserver/gpu/vllm-omni-edit/base/            # decode Deployment + SA overlay
modelserver/gpu/vllm-omni-edit/gke/             # GKE overlay
```

Pool isolation: label `llm-d.ai/guide: optimized-baseline-edit` + own namespace
(`llm-d-edit`).

## Deploy

```bash
# from the repo root
export REPO_ROOT=$(pwd)
export NAMESPACE=llm-d-edit
export GUIDE_DIR=$REPO_ROOT/guides/diffusion-serving/k8s/llm-d-optimized-baseline-edit
source $REPO_ROOT/guides/env.sh   # ROUTER_* vars

kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 1. Router (standalone mode)
helm install llm-d-edit $ROUTER_STANDALONE_CHART \
  -f $REPO_ROOT/guides/recipes/router/base.values.yaml \
  -f $GUIDE_DIR/router/edit.values.yaml \
  -n $NAMESPACE --version $ROUTER_CHART_VERSION

# 2. Model server (GKE overlay; use base/ off-GKE)
kubectl apply -n $NAMESPACE -k $GUIDE_DIR/modelserver/gpu/vllm-omni-edit/gke/
```

> After changing `router/edit.values.yaml`: `helm upgrade` (same flags) AND
> `kubectl rollout restart deploy/llm-d-edit-epp -n $NAMESPACE`.

## Verify

```bash
# Terminal 1 — leave running
kubectl port-forward -n $NAMESPACE svc/llm-d-edit-epp 8082:80
```

```bash
# Terminal 2 — edit a public image through the router, save locally.
# (Input can be an https URL or a data:image/...;base64 URL.)
curl -s --max-time 600 -X POST http://localhost:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role":"user","content":[
      {"type":"text","text":"Change the background to a snowy mountain scene."},
      {"type":"image_url","image_url":{"url":"https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}}
    ]}],
    "extra_body": {"num_inference_steps": 30, "guidance_scale": 5.0, "guidance_scale_2": 2.0, "seed": 42}
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d, -f2 | base64 -d > edited.png

file edited.png     # -> PNG image data
```

Response shape: the edited image comes back as a **data URL inside the message
content** (`choices[0].message.content[0].image_url.url`) — same as the
image-edit examples in the vllm-omni repo.

**Verified working 2026-07-02:** `HTTP 200`, 1.2 MB PNG returned end-to-end
through the EPP router (20 steps, snowy-mountain background edit of a public
image) with 2 decode replicas Ready.
