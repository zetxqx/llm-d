# How diffusion inference works: a request-to-response walkthrough

An educational walkthrough of text-to-image diffusion serving, using the exact
model and stack we deploy: **Z-Image-Turbo on vLLM-Omni** (the
`llm-d-omni` stack). Every step is anchored to real code in the `vllm-omni/`
clone so you can read along. Written for someone who knows LLM serving
(vLLM, KV cache, continuous batching) and wants the diffusion equivalent
mental model.

**The one-paragraph version:** an LLM builds its output one token at a time,
left to right. A diffusion model instead starts from a canvas of **pure random
noise** and runs the *same* denoising network over the *whole* canvas N times
(e.g. 20–50 "steps"), each pass removing a little noise while steering toward
the text prompt, until a clean image remains. Nothing is generated
sequentially, there is no KV cache to reuse, and the total compute is fixed
the moment the request arrives: `steps × canvas size`. That single fact drives
everything unusual about serving these models.

---

## 0. The cast of components

A text-to-image "model" like Z-Image-Turbo is actually **three neural networks
plus one tiny algorithm**, shipped together in one checkpoint (as subfolders —
`text_encoder/`, `transformer/`, `vae/`, `scheduler/` — loaded in
`vllm_omni/diffusion/models/z_image/pipeline_z_image.py:176-209`):

| Component | What it is | Z-Image specifics | Runs |
| --- | --- | --- | --- |
| **Text encoder** | An LLM used *only* to read the prompt and produce embeddings — it generates nothing | A Qwen-family causal LM; embeddings taken from its second-to-last hidden layer, 2560-dim, max 512 prompt tokens | **once** per request |
| **DiT** (Diffusion Transformer) | The denoiser — the big network, where ~all the compute goes. Predicts "what noise is in this image right now" | `ZImageTransformer2DModel`: 30 transformer blocks, hidden dim 3840, 30 heads (`z_image_transformer.py:593-790`) | **once or twice per step × N steps** |
| **VAE** (Variational Autoencoder) | A compressor/decompressor between pixel space and a small "latent" space. Diffusion never touches pixels — it works on the compressed version | `DistributedAutoencoderKL`; 16-channel latents at 1/16 the resolution per side | decode **once** at the end (encode too, for image editing) |
| **Scheduler** | Not a neural net — a short numerical recipe deciding how much noise to remove per step and how to update the canvas | `FlowMatchEulerDiscreteScheduler` (flow matching, the modern formulation) | one cheap update per step |

Two vocabulary items worth internalizing:

- **Latent space:** the VAE compresses a 1024×1024×3 image into a
  64×64×16 tensor — 48× fewer values. All denoising happens there. This is
  why diffusion is feasible at all: the DiT attends over ~1–4k latent tokens,
  not a million pixels.
- **A "step":** one full forward pass of the DiT over the entire latent
  canvas. Steps are strictly sequential — step *i+1* consumes step *i*'s
  output. You cannot parallelize across steps (only within one, by sharding
  the canvas — that's what xDiT/PipeFusion do).

## 1. The request arrives

Our smoke-test request (what `k8s/smoke-test.sh` sends through the llm-d
router):

```json
POST /v1/images/generations
{
  "model": "Tongyi-MAI/Z-Image-Turbo",
  "prompt": "a corgi wearing sunglasses on a beach",
  "size": "1024x1024",
  "n": 1
}
```

vLLM-Omni's API server handles this route in
`vllm_omni/entrypoints/openai/api_server.py:1709-1812`. It parses the OpenAI
fields plus diffusion-specific extras the OpenAI API never had:
`num_inference_steps`, `guidance_scale`, `negative_prompt`, `seed`,
`flow_shift` (`protocol/images.py:33`). It builds an `OmniTextPrompt` and an
`OmniDiffusionSamplingParams` and submits them to the `AsyncOmni` orchestrator
(`entrypoints/async_omni.py:249-398`) — the omni counterpart of vLLM's
`AsyncLLMEngine`, which resolves per-stage sampling params and enqueues the
request into the diffusion engine.

> **Note the routing significance:** the request body *declares the total
> compute* — `num_inference_steps × (width × height)`. No LLM request does
> that (output length is unknown). This is the foundation of the Tier-2
> "declared-cost routing" idea in
> [routing-opportunities.md](routing-opportunities.md).

## 2. Text encoding — the prompt becomes numbers (once)

`ZImagePipeline.encode_prompt()` (`pipeline_z_image.py:241-324`) wraps the
prompt in a chat template, tokenizes it (padded to 512), and runs the Qwen
text encoder a single time. The output is a `[prompt_len × 2560]` embedding
matrix — the "meaning" of your prompt, which the DiT will consult at every
step. If a `negative_prompt` was given (things to avoid), it's encoded the
same way; otherwise an empty string is encoded.

This is the only part of the pipeline that resembles LLM inference — one
prefill-like pass, no decode. It's typically <5% of request time. (In
disaggregated E/DiT/VAE serving — vLLM-Omni RFC #4590 — this becomes its own
stage/pool.)

## 3. Latent initialization — the canvas of pure noise

`prepare_latents()` (`pipeline_z_image.py:326-381`) creates the starting
point: a tensor of Gaussian random noise with shape

```
[batch, 16 channels, height/16, width/16]   → for 1024×1024: [1, 16, 64, 64]
```

via `randn_tensor()` seeded by the request's `seed` (this is why the same
seed + prompt reproduces the same image). At this moment the "image" is 100%
static — like a TV with no signal. Everything from here on is subtraction of
noise, never addition of content.

The scheduler is then configured with the timestep sequence: N values going
from "fully noisy" to "clean", spaced by the flow-matching schedule (with a
resolution-dependent shift `mu`, `pipeline_z_image.py:700-716`). The pipeline
default is `num_inference_steps=50` (`:419`), but **"Turbo" means
step-distilled** — the model was trained to reach good quality in far fewer
steps, so real deployments pass a small `num_inference_steps` (single digits)
and get near-interactive latency. Steps are the #1 knob trading quality for
latency, linearly.

## 4. The denoising loop — where all the time goes

The heart of the pipeline (`pipeline_z_image.py:732-821`), annotated:

```python
for i, t in enumerate(timesteps):                      # N sequential steps
    # (a) Classifier-free guidance: duplicate the canvas so the DiT
    #     sees it twice — once with the prompt, once with the negative/
    #     empty prompt.
    latent_model_input = latents.repeat(2, 1, 1, 1)            # :758

    # (b) THE expensive call: full DiT forward over all latent tokens
    model_out = self.transformer(latent_model_input,
                                 timestep, prompt_embeds)      # :769-773

    # (c) Combine the two predictions: amplify what the prompt adds
    pred = pos + guidance_scale * (pos - neg)                  # :785

    # (d) Cheap numerical update: remove one step's worth of noise
    latents = self.scheduler.step(pred, t, latents)[0]         # :809
```

Unpacking the two ideas inside:

**What the DiT actually computes.** Inside
`ZImageTransformer2DModel.forward()` (`z_image_transformer.py:927-1030`), the
64×64 latent canvas is **patchified** — cut into 2×2 patches, each becoming
one token (`x_embedder`, `:712-718`). So a 1024×1024 image is a sequence of
32×32 = **1024 latent tokens**, which are concatenated with the prompt's text
tokens and run through 30 transformer blocks of joint self-attention (every
image patch attends to every other patch *and* to the prompt). The output is
un-patchified back into a canvas-shaped tensor: the model's estimate of the
noise/velocity at this timestep. Given the timestep as input, the *same
weights* serve every step — a step is not a layer; it's a repeat invocation.

**What CFG (classifier-free guidance) is.** The model makes two predictions —
"denoise toward the prompt" and "denoise toward anything" — and the update
follows the *difference*, scaled by `guidance_scale` (pipeline default 5.0,
`:421`). It's contrast enhancement for prompt adherence, and it **doubles the
DiT compute of every step** (the `repeat(2,...)` above). Turbo-distilled
models often bake guidance in and run with CFG effectively off — one more
request parameter that halves or doubles the work.

So the cost of a request, exactly:

```
total compute ≈ N_steps × (2 if CFG else 1) × DiT_forward(seq_len ∝ width×height / 1024)
```

Every term is known before the first FLOP. For an LLM, the equivalent formula
contains the unknowable `output_length`. This determinism is diffusion's
defining serving property.

## 5. VAE decode — latents become pixels (once)

After the last step, the latents are a clean image *in compressed form*. One
call to `vae.decode()` (`pipeline_z_image.py:825-828`) upsamples 64×64×16 →
1024×1024×3 through the VAE's convolutional decoder. It's a single pass but
memory-hungry at high resolution (activations scale with pixel count) — this
is what `--vae-use-slicing` / `--vae-use-tiling` mitigate by decoding in
chunks, and why RFC #4590 reports VAE decode dominating for long *video*
(many frames to decode). For 1024² stills it's a small fraction of the
request.

## 6. Response — tensor to PNG to JSON

The image tensor is converted to a PIL image (`VaeImageProcessor.postprocess`,
`pipeline_z_image.py:239`), PNG-encoded, base64'd
(`api_server.py:1810`, `image_api_utils.py:56-68`), and wrapped in the OpenAI
response shape:

```json
{ "created": 1751700000,
  "data": [ { "b64_json": "iVBORw0KGgo..." } ] }
```

That base64 blob is the multi-hundred-KB body our smoke test pipes through
`jq -r '.data[0].b64_json' | base64 -d > out.png`. (Dynamo's alternative:
write to object storage and return a URL — see
[dynamo-diffusion-omni-report.md](dynamo-diffusion-omni-report.md), Step 7.)

## 7. Timeline and the serving consequences

For one 1024² request at ~9 steps with CFG, qualitatively:

```
|txt enc|■■■■■■■■■■■■■■■■■■  DiT: N × (2×forward + sched)  ■■■■■■■■■■■■■■■■■■|VAE|png|
  ~few %                        ~90%+ of wall time                       small  tiny
```

Side by side with the LLM serving model you know:

| | LLM (vLLM) | Diffusion (vLLM-Omni) |
| --- | --- | --- |
| Generation | autoregressive, token by token | iterative refinement of the whole canvas |
| Cost known at arrival? | no (output length unknown) | **yes**: steps × resolution × CFG |
| KV cache | central (prefix reuse, paged memory) | **none** — no tokens, nothing to prefix-match |
| Batching | continuous batching, requests join/leave per token | **none across requests** in vLLM-Omni today: FIFO `RequestScheduler`, `max_num_running_reqs=1` — one request owns the GPU for its full loop (`diffusion/diffusion_engine.py:141-144`); only `n>1` images of the *same* request batch together |
| Streaming | token stream | nothing to stream until the end (previews of intermediate latents are possible but not typical) |
| Memory pressure | grows with context/KV | flat weights + activations; peaks at startup and VAE decode (our L4 OOM → `--enable-cpu-offload`; `--enable-layerwise-offload` pages DiT blocks per step, `z_image_transformer.py:616`) |
| Useful engine metrics | `num_waiting`, `kv_cache_usage` | queue depth + **steps remaining** (deterministic time-to-free) — not exported today |

The routing morals, in one breath: with no KV cache there is nothing for
prefix-affinity to do; with no cross-request batching, a busy worker means a
request **waits the full remaining loop** of the one ahead of it, so queue
position × declared cost is a *precise* wait-time prediction — better
information than any LLM router ever gets, and currently used by no router
(llm-d, Dynamo, SMG all included). That is the opportunity documented in
[routing-opportunities.md](routing-opportunities.md).

## Appendix: mapping to the other modalities we run

- **OmniGen2 (image editing, `llm-d-edit`):** same loop, two differences —
  the input image is VAE-*encoded* into latents that condition the denoising
  (instead of starting from pure noise semantics), and its text encoder is a
  full multimodal LLM (the startup-OOM culprit on L4).
- **Qwen3-TTS (`llm-d-tts`):** *not* diffusion — an autoregressive
  token-by-token model like an LLM (hence it has a prefix cache and can
  stream audio). The contrast between our TTS and image stacks is exactly the
  AR-vs-diffusion contrast in the table above.
- **Video:** diffusion with a time axis — latents gain a frame dimension,
  sequence length multiplies by frames, and VAE decode becomes a major cost
  (RFC #4590's motivation for a dedicated VAE pool).
