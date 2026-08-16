# llm-d optimized-baseline for vLLM-Omni Qwen3-TTS

An llm-d stack serving **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`,
text → speech via `POST /v1/audio/speech`) behind the llm-d Router /
EndpointPicker. Companion to the diffusion stack in
[`../llm-d-optimized-baseline-omni/`](../llm-d-optimized-baseline-omni/README.md)
— together they cover the two serving regimes of vLLM-Omni:

| | Diffusion stack (Z-Image-Turbo) | **This stack (Qwen3-TTS)** |
| --- | --- | --- |
| Generation | non-autoregressive DiT loop | **autoregressive**, 2-stage (talker → code2wav) |
| KV / prefix cache | none | **yes — talker runs `enable_prefix_caching: true`** |
| API | `/v1/images/generations` | `/v1/audio/speech` (+ `/stream`, `/batch`, `/v1/audio/voices`) |
| Output | PNG (base64 JSON) | raw WAV bytes |
| GPUs / replica | 1 | 1 (both stages pinned to device 0; codec chunks stream between stages via SharedMemoryConnector → `/dev/shm` is hot-path) |

## Why this is interesting for the routing investigation

TTS is the **middle case** between LLM and diffusion routing:

- Like diffusion, the stock EPP profile fails on the path: no parser is
  registered for `/v1/audio/speech`, so requests would 400 exactly as
  `/v1/images/generations` did (verified live 2026-07-02 on the diffusion
  stack). Hence `payload-agnostic.yaml` here too.
- **Unlike** diffusion, the backend has a real AR KV/prefix cache (talker stage,
  see [`vllm_omni/deploy/qwen3_tts.yaml`](../../vllm-omni/vllm_omni/deploy/qwen3_tts.yaml)
  — prefix caching is a net win for repeated text/voice prefixes). If the omni
  server aggregates per-stage `vllm:*` metrics on `/metrics`, the stock
  queue/kv-cache/prefix scorers could actually light up for TTS. **Open
  experiment:** scrape `/metrics` on a decode pod under load and check.

## Structure

```
router/tts.values.yaml                          # EPP config (payload-agnostic)
modelserver/gpu/vllm-omni-tts/base/             # decode Deployment + SA overlay
modelserver/gpu/vllm-omni-tts/gke/              # GKE overlay
```

Pool isolation: pods are labeled `llm-d.ai/guide: optimized-baseline-tts`
(distinct from the diffusion stack's `optimized-baseline`) so the two
InferencePools can never select each other's endpoints, and the stack deploys
into its own namespace (`llm-d-tts`).

## Deploy

```bash
# from the workspace root
export WORKSPACE=$(pwd)
export NAMESPACE=llm-d-tts
export GUIDE_DIR=$WORKSPACE/k8s/llm-d-optimized-baseline-tts
source $WORKSPACE/llm-d/guides/env.sh   # ROUTER_* vars

kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 1. Router (standalone mode)
helm install llm-d-tts $ROUTER_STANDALONE_CHART \
  -f $WORKSPACE/llm-d/guides/recipes/router/base.values.yaml \
  -f $GUIDE_DIR/router/tts.values.yaml \
  -n $NAMESPACE --version $ROUTER_CHART_VERSION

# 2. Model server (GKE overlay; use base/ off-GKE)
kubectl apply -n $NAMESPACE -k $GUIDE_DIR/modelserver/gpu/vllm-omni-tts/gke/
```

> After any change to `router/tts.values.yaml`: `helm upgrade` (same flags) AND
> `kubectl rollout restart deploy/llm-d-tts-epp -n $NAMESPACE` — the EPP reads
> its plugin ConfigMap at startup only.

## Verify

```bash
# Terminal 1 — leave running
kubectl port-forward -n $NAMESPACE svc/llm-d-tts-epp 8081:80
```

```bash
# Terminal 2 — synthesize speech through the router, save locally
curl -s -X POST http://localhost:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello from the llm-d routed text to speech stack.","voice":"vivian","response_format":"wav"}' \
  --output out.wav

file out.wav        # -> RIFF ... WAVE audio
# Listen: open out.wav (macOS) / xdg-open out.wav (Linux)
```

Note the response is **raw WAV bytes**, not JSON — no `jq`/`base64` step.

The CustomVoice model rejects `voice: "default"` (HTTP 400). Its preset voices
(from the server's error message, verified live):
`aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian`.

**Verified working 2026-07-02:** `HTTP 200`, 134 KB RIFF/WAV returned end-to-end
through the EPP router with 2 decode replicas Ready (~3.5 min rollout).

### Via `/v1/chat/completions`

vLLM-Omni also serves TTS models through the OpenAI chat API: the text to
synthesize goes in as a user message, and `voice` / `language` are top-level
request fields (not `extra_body` — unlike the diffusion stack's image params).
The audio comes back in the OpenAI audio-output shape: base64 **WAV** in
`choices[0].message.audio.data`. Since the router uses the passthrough parser,
this path routes through the EPP the same way `/v1/audio/speech` does.

```bash
ƒ

file out.wav        # -> RIFF ... WAVE audio
```

`"modalities": ["audio"]` is optional here — it defaults to the model's
declared output modality (audio for Qwen3-TTS) — but explicit is clearer.
Unlike `/v1/audio/speech` there is no `response_format` choice on this path:
the server always returns base64 WAV inside JSON, so prefer `/v1/audio/speech`
for streaming/PCM. (Chat-path example derived from
`vllm_omni/entrypoints/openai/serving_chat.py`; not yet verified live through
this stack.)

## Streaming (play audio on the fly)

Same endpoint, two extra fields: `"stream": true` + `"response_format": "pcm"`.
The server then streams **raw 24 kHz mono int16 PCM** over chunked HTTP as
synthesis proceeds. (WAV can't stream — its header needs a known total length.)
The deploy config is already tuned for this: `async_chunk: true` and
`initial_codec_chunk_frames: 1` emit the first audio chunk as early as
possible to minimize time-to-first-audio (TTFA).

```bash
# Terminal 1 — leave running
kubectl port-forward -n $NAMESPACE svc/llm-d-tts-epp 8081:80
```

```bash
# Terminal 2 — synthesize AND play while it generates (curl -N = no buffering)
curl -sN -X POST http://localhost:8081/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello! This audio is playing while it is still being generated.","voice":"vivian","response_format":"pcm","stream":true}' \
  | aplay -q -f S16_LE -r 24000 -c 1 -

# Player alternatives:
#   | ffplay -autoexit -nodisp -loglevel error -f s16le -ar 24000 -ac 1 -
#   | pw-play --rate 24000 --format s16 --channels 1 -
```

Richer clients in the vllm-omni repo
([`examples/online_serving/text_to_speech/qwen3_tts/`](../../vllm-omni/examples/online_serving/text_to_speech/qwen3_tts/)):
`streaming_speech_client.py` (Python streaming client, saves WAV),
`gradio_demo.py` (browser playback; WebSocket + FastRTC variants),
`word_timestamps_demo.py` (word-level timestamps alongside the stream).

**Streaming through the router verified 2026-07-02:** response headers in
~13 ms, then ~1 MB PCM (≈ 20.8 s of audio) delivered incrementally on the same
connection — the EPP's Envoy sidecar does **not** buffer the response, so
on-the-fly playback works behind llm-d routing.

Routing note: a streaming request occupies its endpoint for the full utterance
duration, and the user-facing metric becomes **TTFA**, not request latency —
`active-request-scorer` (in-flight = still-streaming) models this load
correctly, and TTFA p99 through-router vs direct is a good benchmark metric
for the routing investigation.
