# TTS vs text-to-image: two different model families, two different serving physics

Text-to-speech and text-to-image are built on different generation
architectures — **autoregressive (AR)** vs **diffusion** — and nearly every
serving/routing property we care about follows from that split. This doc
pins down the distinction, the exceptions, and the consequences, using the
models we actually run: Qwen3-TTS (`llm-d-tts`), Z-Image-Turbo
(`llm-d-omni`), OmniGen2 (`llm-d-edit`).

Companions: [diffusion-inference-walkthrough.md](diffusion-inference-walkthrough.md)
(inside the diffusion loop), [vllm-omni-stage-pipeline.md](vllm-omni-stage-pipeline.md)
(how stages compose into pipelines).

## The typical pairing

**Text-to-image → diffusion.** Z-Image, Qwen-Image, SD3, Flux: text encoder →
DiT iteratively denoises a whole latent canvas (N steps over ~1–4k latent
tokens) → VAE decodes to pixels. The image is generated *all at once,
refined globally* — no part of it exists "first."

**Text-to-speech → autoregressive.** Qwen3-TTS, the Qwen3-Omni Talker: an
LLM-like transformer generates **audio codec tokens** one at a time, left to
right — exactly like text generation, just over a vocabulary of quantized
sound units (RVQ codes) instead of words. A **vocoder** (Code2Wav) then
converts codec tokens to waveform — but that is a cheap parallel decoder
(the VAE's role, not the DiT's); there is no denoising loop in this path.

## Why each modality gravitates to its architecture

The data's shape picks the architecture:

- **Audio is 1-D and consumed sequentially in time.** Listeners consume it
  front-to-back, so generating front-to-back is natural — and it is what
  makes **streaming** possible (our 13 ms TTFB through the llm-d router):
  second 1 can play while second 5 is generated. Prosody also depends on
  what was already said — long-range sequential order, AR's home turf.
- **An image is 2-D and consumed holistically.** There is no meaningful
  "first pixel"; sky and ground must be consistent from the start.
  Diffusion's global-refinement loop matches: every step updates the whole
  canvas coherently. (Images *can* be generated AR patch-by-patch — early
  DALL-E, Parti — but quality/efficiency pushed the field to diffusion.)

## Exceptions worth knowing

- **Hybrids are common in the omni world.** GLM-Image (the model behind
  Dynamo's disaggregated omni example) is **AR → DiT**: an autoregressive
  stage plans the image, a diffusion stage renders it. Mammoth-MoDA in the
  vllm-omni stage configs is similar. "Image model" can span both regimes
  across its stages.
- **TTS is not always pure-AR.** Some modern TTS systems (CosyVoice-,
  F5-TTS-style) use **flow-matching/diffusion** for the acoustic model,
  trading streamability for quality. Qwen3-TTS and the Qwen3-Omni Talker
  are AR — which is exactly why they stream.
- **Video = diffusion with a time axis** (Wan, CogVideoX), despite being
  temporal like audio — per-frame spatial coherence dominates, and VAE
  decode becomes a major cost (RFC #4590).

## The serving-physics table

Every row below is a direct consequence of AR vs diffusion:

| Property | TTS (AR) | Text-to-image (diffusion) |
| --- | --- | --- |
| Generation | token-by-token, sequential | whole canvas, N refinement steps |
| KV / prefix cache | **yes** — real cache-affinity signal (further boosted once RFC #1184 fixes hidden-state I/O + prefix caching) | **none** — no tokens, nothing to prefix-match |
| Streaming | yes; TTFA/TTFB is the latency metric | no; all-or-nothing delivery (E2E latency is the metric) |
| Cost known at arrival | no — output length unknown, like an LLM | **yes** — steps × resolution × (CFG?) declared in the body |
| Batching (vLLM-Omni today) | continuous batching | none across requests: FIFO, `max_num_running_reqs=1` |
| Queue behavior | requests interleave; marginal slowdown | head-of-line blocking: wait = full remaining loops ahead of you |
| Right routing signal | prefix/session affinity + queue depth | declared cost / work-remaining |
| Intra-request accel | speculative decoding etc. | step caching (TeaCache/Cache-DiT), step distillation ("Turbo") |

## Why this is the backbone of the routing story

- A single routing policy — token-denominated, cache-affine, as shipped by
  llm-d defaults, Dynamo's KV router, and SMG alike — matches only the AR
  column. The diffusion column needs a different signal entirely (declared
  cost), and it is *available at routing time*, which it never is for AR.
- Our three stacks are the argument in miniature: `llm-d-tts` (AR: affinity
  + queue), `llm-d-omni` (diffusion: cost), `llm-d-edit` (diffusion +
  multi-MB input bodies).
- Hybrids are the strongest case: one GLM-Image request crosses *both*
  regimes, so disaggregated per-stage pools genuinely need per-stage routing
  policies — the [routing-opportunities.md](routing-opportunities.md) Tier-4
  claim.
