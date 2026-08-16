# vLLM-Omni model server — multimodal capabilities

What the vLLM-Omni model server can do, and how to invoke each capability over
the OpenAI-compatible API. Two layers:

1. **Framework capabilities** — everything vLLM-Omni supports (depends on which
   model you load).
2. **Your running servers** — what the two services in the `omni` namespace
   actually expose today.

> Source: the cloned `vllm-omni` repo (`README.md`, `vllm_omni/deploy/*.yaml`,
> serving entrypoints) at the version pinned in this workspace. Capabilities are
> **per loaded model** — a server only does what its model supports. Always
> confirm with `GET /v1/models` against the specific service.

---

## 1. Framework capability matrix

vLLM-Omni extends vLLM from text-only autoregressive (AR) generation to
**any-to-any** omni-modality, including **non-autoregressive** Diffusion
Transformer (DiT) models. Core modalities: **text, image, video, audio**.

Full request/response shapes are in [§4](#4-request--response-formats); the
"Ref" column links to the runnable example each shape is drawn from.

| Capability | Input → Output | Example models | API endpoint | Ref (req/resp) |
| --- | --- | --- | --- | --- |
| Text chat / completion | text → text | Qwen3-Omni (thinker) | `POST /v1/chat/completions`, `POST /v1/completions` | [omni chat](#omni-chat--multimodal-understanding) |
| Vision understanding | text + image → text | Qwen2.5/3-Omni, MiniCPM-o | `POST /v1/chat/completions` | [omni chat](#omni-chat--multimodal-understanding) |
| Video understanding | text + video → text | Qwen3-Omni | `POST /v1/chat/completions` | [omni chat](#omni-chat--multimodal-understanding) |
| Audio understanding (ASR / audio QA) | text + audio → text | Qwen3-Omni, MiMo-Audio | `POST /v1/chat/completions` | [omni chat](#omni-chat--multimodal-understanding) |
| Omni chat (speech out) | text/image/video/audio → text + audio | Qwen2.5/3-Omni, Ming-Omni | `POST /v1/chat/completions` | [omni chat](#omni-chat--multimodal-understanding) |
| Text-to-speech (TTS) | text → audio | Qwen3-TTS, MOSS-TTS, CosyVoice3, VoxCPM2, IndexTTS2 | `POST /v1/audio/speech` | [TTS](#text-to-speech-tts) |
| Text-to-image (diffusion) | text → image | Z-Image-Turbo, Qwen-Image, FLUX, GLM-Image, HunyuanImage | `POST /v1/images/generations` | [text→image](#text-to-image) |
| Image editing | text + image → image | Bagel, Ming-Flash-Omni-Image | `POST /v1/chat/completions` (or `/v1/images/edits`) | [image edit](#image-editing-image-to-image) |
| Text-to-video (diffusion) | text → video | Wan2.2, DreamZero | `POST /v1/videos` → poll `GET /v1/videos/{id}` | [text→video](#text-to-video-async-job) |
| World models | text/image → video | NVIDIA Cosmos3, DreamZero, Gr00t-N1 | `POST /v1/videos` | [text→video](#text-to-video-async-job) |
| Embeddings | text/mm → vector | (model-dependent) | `POST /v1/embeddings` | OpenAI embeddings schema |

> Audio/vision/video **understanding** is done by passing those inputs *inside*
> a `chat/completions` message (multimodal content parts) — there is no separate
> transcription route in this build. **Generation** of non-text output uses the
> dedicated `audio/speech`, `images/*`, and `videos` routes. Note image editing
> in the shipped examples goes through `chat/completions` (image in, image back
> as a data URL), even though an `/v1/images/edits` route also exists.

### Architectural notes (relevant to the llm-d investigation)
- **AR vs DiT**: AR (LLM/omni-chat/TTS thinker) stages use vLLM KV-cache. DiT
  (image/video diffusion) stages do **not** — no token KV cache, no prompt
  prefix cache. This is why the optimized-baseline cache-aware router scorers
  don't apply to the diffusion services (see
  [`../k8s/llm-d-optimized-baseline-omni/README.md`](../k8s/llm-d-optimized-baseline-omni/README.md)).
- **Multi-stage pipelines**: omni models run as a pipeline of stages (e.g.
  Qwen2.5-Omni = thinker → talker → code2wav across GPUs). Stages are declared in
  `vllm_omni/deploy/<model>.yaml` and can be disaggregated.

---

## 2. Endpoints exposed by the server

Confirmed in the serving entrypoints (availability still depends on the model):

**OpenAI-compatible**
- `GET  /v1/models` — list loaded model id(s)
- `GET  /health` — liveness
- `POST /v1/chat/completions` — text + multimodal-input chat (optionally audio out)
- `POST /v1/completions` — raw text completion
- `POST /v1/embeddings` — embeddings
- `POST /v1/audio/speech` — TTS (text → audio)
- `POST /v1/images/generations` — text → image
- `POST /v1/images/edits` — image + text → image

**Omni extensions**
- `POST /v1/videos`, `GET /v1/videos/{id}`, `GET /v1/videos/{id}/content` — async video generation + retrieval
- `POST /v1/streaming/persona`, `POST /v1/streaming/reset` — streaming persona/session control
- `POST /v1/omni/sleep`, `POST /v1/omni/wakeup` — release / reload model from GPU memory

---

## 3. Your running servers (`omni` namespace)

> The `vllm-omni` service (Qwen2.5-Omni-3B omni chat) that used to be listed
> here was removed 2026-07-15.

### `vllm-omni-image` — Z-Image-Turbo (diffusion text-to-image)
Input: text. Output: **image** (base64 PNG).

```bash
kubectl port-forward -n omni svc/vllm-omni-image 8001:8000
curl -s http://localhost:8001/v1/images/generations -H 'Content-Type: application/json' -d '{
  "model": "Tongyi-MAI/Z-Image-Turbo",
  "prompt": "a dragon over the Green Mountains of Vermont, golden hour",
  "size": "1024x1024", "seed": 42
}' | jq -r '.data[0].b64_json' | base64 -d > out.png
```

> To discover exactly what each server accepts, hit `GET /v1/models` first and
> use the returned `id` as the `model` field — name mismatches are the most
> common cause of 4xx errors.

---

## 4. Request / response formats

Each block below shows the wire shape drawn from a runnable example in the
cloned `vllm-omni` repo. Follow the link for the full script.

### Text-to-image
Endpoint: `POST /v1/images/generations` ·
Ref: [`examples/online_serving/text_to_image/run_curl_text_to_image.sh`](../vllm-omni/examples/online_serving/text_to_image/run_curl_text_to_image.sh)

Request:
```json
{ "prompt": "a dragon over the Green Mountains of Vermont", "size": "1024x1024", "seed": 42 }
```
Response — base64 image under `data[]` (OpenAI Images shape):
```json
{ "data": [ { "b64_json": "<base64-png>" } ] }
```
Extract: `jq -r '.data[0].b64_json' | base64 -d > out.png`

### Image editing (image-to-image)
Endpoint: `POST /v1/chat/completions` (image in / image out) ·
Ref: [`examples/online_serving/image_to_image/run_curl_image_edit.sh`](../vllm-omni/examples/online_serving/image_to_image/run_curl_image_edit.sh)

Request — input image as a data URL, diffusion knobs in `extra_body`:
```json
{
  "messages": [{ "role": "user", "content": [
    { "type": "text", "text": "make it snowy" },
    { "type": "image_url", "image_url": { "url": "data:image/png;base64,<b64>" } }
  ]}],
  "extra_body": { "num_inference_steps": 50, "guidance_scale": 1, "seed": 42 }
}
```
Response — edited image returned as a data URL inside the message content:
```json
{ "choices": [ { "message": { "content": [ { "image_url": { "url": "data:image/png;base64,<b64>" } } ] } } ] }
```
Extract: `jq -r '.choices[0].message.content[0].image_url.url' | cut -d, -f2 | base64 -d`

### Text-to-video (async job)
Endpoints: `POST /v1/videos` → `GET /v1/videos/{id}` (poll) → `GET /v1/videos/{id}/content` ·
Ref: [`examples/online_serving/text_to_video/run_curl_text_to_video.sh`](../vllm-omni/examples/online_serving/text_to_video/run_curl_text_to_video.sh)

Request — **multipart form fields** (`-F`), not JSON:
```
prompt=Two cats boxing on a spotlighted stage.
seconds=2  size=832x480  fps=16  num_inference_steps=40
guidance_scale=4.0  seed=42   negative_prompt=...
```
Create response — a job handle:
```json
{ "id": "<video_id>", "status": "queued" }
```
Poll response — `status` moves `queued` → `in_progress` → `completed` | `failed`.
Then fetch bytes: `GET /v1/videos/{id}/content -o out.mp4`.

### Text-to-speech (TTS)
Endpoint: `POST /v1/audio/speech` (also `/stream`, `/batch`; voice upload via `/v1/audio/voices`) ·
Ref: [`examples/online_serving/text_to_speech/README.md`](../vllm-omni/examples/online_serving/text_to_speech/README.md)

Request (OpenAI speech shape; voice-clone models also take `ref_audio` + `ref_text`):
```json
{ "input": "Hello, how are you?", "voice": "default", "response_format": "wav" }
```
Response — raw audio bytes (not JSON): `curl ... --output output.wav`.

### Omni chat / multimodal understanding
Endpoint: `POST /v1/chat/completions` ·
Ref: [`examples/online_serving/qwen2_5_omni/run_curl_multimodal_generation.sh`](../vllm-omni/examples/online_serving/qwen2_5_omni/run_curl_multimodal_generation.sh)

Request — mix `text` / `image_url` / `audio_url` / `video_url` content parts.
Omni-specific extras: `sampling_params_list` (per pipeline stage),
`mm_processor_kwargs` (e.g. `use_audio_in_video`), `modalities`:
```json
{
  "model": "Qwen/Qwen2.5-Omni-7B",
  "sampling_params_list": [ { "temperature": 0.0 }, { "temperature": 0.9 }, { "temperature": 0.0 } ],
  "mm_processor_kwargs": { "use_audio_in_video": true },
  "modalities": null,
  "messages": [
    { "role": "system", "content": [ { "type": "text", "text": "You are Qwen ..." } ] },
    { "role": "user", "content": [
      { "type": "audio_url", "audio_url": { "url": "https://.../mary_had_lamb.ogg" } },
      { "type": "image_url", "image_url": { "url": "https://.../cherry_blossom.jpg" } },
      { "type": "text", "text": "What is recited? Describe the image." }
    ]}
  ]
}
```
Response — text under `choices[0].message.content`; **generated speech** is
returned as (large) binary audio in the message content when the talker/code2wav
stages are active (the example prints only the text to keep output readable).

### Broader per-modality examples
The cloned repo has a runnable client per modality under
[`examples/online_serving/`](../vllm-omni/examples/online_serving/) —
including `text_to_image`, `image_to_image`, `image_to_video`,
`speech_to_video`, `streaming_video_generation`, `text_to_speech/<model>`,
`qwen3_omni`, and the combined
[`openai_chat_completion_client_for_multimodal_generation.py`](../vllm-omni/examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py).
