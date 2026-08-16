# Realtime voice interaction with vLLM-Omni — findings & reproduction guide

How to talk to omni models running on the `bobbm` GKE cluster — with your
voice, getting spoken replies — and what we learned bringing it up. Everything
here was **executed and verified live** on 2026-07-03/04; each section links
to the folder that holds the runnable pieces.

> Companion docs: [vllm-omni-capabilities.md](vllm-omni-capabilities.md) (the
> full API surface), [routing-opportunities.md](routing-opportunities.md)
> (llm-d scheduling angles).

---

## 1. The core finding: two voice regimes, and a hard model constraint

vLLM-Omni offers two ways to have a voice conversation, and they are **not**
interchangeable:

| | Turn-based voice chat | True realtime streaming |
| --- | --- | --- |
| API | `POST /v1/chat/completions`, `modalities: ["text","audio"]`, `stream: true` | `GET /v1/realtime` WebSocket |
| Interaction feel | walkie-talkie: record → wait → listen | audio streams *while* you speak; reply plays *while* it generates |
| Models | any `--omni` pipeline with audio out (Qwen2.5-Omni, Qwen3-Omni, …) | **Qwen3-Omni ONLY** |
| Smallest GPU footprint | Qwen2.5-Omni-3B on 2× L4 (setup removed 2026-07-15) | Qwen3-Omni-30B on 2× H100-80G (deployed) |

**Why realtime is Qwen3-Omni-only** (verified 2026-07-03): the realtime input
path calls `buffer_realtime_audio` on the model class, and the sole
implementer is
`vllm-omni/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py` — in
both the deployed `vllm/vllm-omni:v0.22.0` image and the repo at HEAD.
Against Qwen2.5-Omni the WebSocket *handshake succeeds* (`101 Switching
Protocols` — misleading!) and then the first commit fails:

```
{'type': 'error', 'error': "type object 'Qwen2_5OmniForConditionalGeneration'
 has no attribute 'buffer_realtime_audio'", 'code': 'processing_error'}
```

A second non-obvious API fact (verified 2026-07-03): on the chat-completions
path, **reply audio only exists in streaming mode**. A non-streaming request
returns `message.audio: null`; with `stream: true` the audio arrives as SSE
chunks tagged `modality: "audio"` whose concatenated `delta.content` decodes
to one base64 WAV (24 kHz mono PCM16).

---

## 2. What is deployed where

| Service | Namespace | Model | GPUs | Status |
| --- | --- | --- | --- | --- |
| `optimized-baseline-realtime-...-decode` (+ llm-d router + `...-direct` Service) | `llm-d-realtime` | Qwen3-Omni-30B-A3B-Instruct | 2× H100 (spot, `bobbm-spoth100` pool) | verified 2026-07-14 |
| `vllm-omni-image` | `omni` | Z-Image-Turbo | 1× L4 | — |

Realtime now lives in ONE deployment (the llm-d stack in [`../realtime/`](../realtime/README.md))
with two access paths: the llm-d router and a plain k8s `-direct` Service
over the same pods. Earlier separate setups were removed: the Qwen2.5-Omni-3B
turn-based chat service + `realtime-playground/` folder (2026-07-15), and the
standalone `vllm-omni-realtime` deployment in `omni` + its
router-in-front experiment + `realtime/standalone-h100/` folder (2026-07-15).

Cost note: the H100 deployment keeps a spot node alive. Park it with
`kubectl scale -n llm-d-realtime deploy/optimized-baseline-realtime-nvidia-gpu-vllm-decode --replicas=0`
(the pool autoscales back down); re-scale to 1 to resume (~30-40 min to Ready
— see §3 timings).

---

## 3. Reproduce: true realtime (via the direct Service — works today)

Folder: [`../realtime/`](../realtime/README.md) — full instructions there.
Condensed (deploy per that README if the stack is not already up):

```bash
kubectl port-forward -n llm-d-realtime \
  svc/optimized-baseline-realtime-nvidia-gpu-vllm-direct 8083:80   # leave running

cd realtime
pip install --index-url https://pypi.org/simple websockets         # once
python3 realtime-mic-client.py            # live mic conversation
python3 realtime-mic-client.py --input-wav input_16k_mono.wav --no-play
```

First-rollout timeline observed on the original standalone variant
(2026-07-04, same image/model): spot node scale-up ~1 min → 9.5 GB image
pull → ~65 GB model download → 3-stage engine init → **Ready at ~38 min**.
(The llm-d stack's 2026-07-14 rollout was faster — ~20 min — the image was
already cached on the node.)

Verification runs (2026-07-04 on the standalone variant, re-verified
2026-07-14 through the direct Service): both the upstream reference client
(`vllm-omni/examples/online_serving/qwen3_omni/openai_realtime_client.py`)
and `realtime-mic-client.py` completed full sessions — streamed
transcription + 24 kHz PCM reply deltas; e.g. `rt_h100_output.wav`
(~287 s of speech: the model recognized the clip as Edison's 1877 recording
and delivered a five-minute history lesson — replies stream, so playback
starts immediately, but budget client timeouts for the tail).

WebSocket protocol in one glance (input must be 16 kHz mono PCM16):

```
-> {"type":"session.update","model":"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
-> {"type":"input_audio_buffer.commit","final":false}
-> {"type":"input_audio_buffer.append","audio":"<b64 pcm16 chunk>"}   (repeat)
-> {"type":"input_audio_buffer.commit","final":true}
<- response.audio.delta {audio: <b64 raw PCM>, sample_rate_hz: 24000}  (repeat)
<- transcription.delta / transcription.done
<- response.audio.done
```

---

## 4. Reproduce: llm-d-routed realtime (deployed 2026-07-14)

Folder: [`../realtime/`](../realtime/README.md) (same deployment as §3 — this is its router path).

Deployed live 2026-07-14 (namespace `llm-d-realtime`, 1 replica). Outcome,
in short — full detail and captured logs in the folder README:

- The router path cannot complete a WebSocket handshake: the EPP schedules
  only after the request body's `EndOfStream`, which an upgrade never sends,
  so the handshake hangs until the client times out.
- The stack therefore also ships a plain k8s Service over the same decode
  pods (`...-direct`); through it the same client completes full realtime
  sessions. That is the working path until the EPP schedules upgrades at
  header time.
- Session balancing across ≥2 replicas remains unexercised (scaled to 1
  replica for GPU budget).

---

## 5. Gotchas log (each cost real time)

1. **`/v1/realtime` handshake success proves nothing** — model support fails
   only at the first commit (§1). Test with an actual audio commit.
2. **No audio in non-streaming chat responses** — always `stream: true` when
   you want speech out (§1).
3. **Corp Airlock pip mirror lacks common OSS packages (e.g. websockets)** — install client deps with
   `--index-url https://pypi.org/simple`. `/etc/pip.conf` is
   airlock-managed and reverts edits.
4. **Boot disk on the H100 pool is 100 GB** — image (9.5 GB) + model
   (~65 GB) barely fit; the Deployment requests `ephemeral-storage: 75Gi` so
   the scheduler accounts for it. Don't co-schedule another model download.
5. **Spot preemption re-downloads the model** (hf-cache is an emptyDir) —
   acceptable for a playground; use a PVC if it gets annoying.
6. **`kubectl port-forward` tunnels die on idle** — if every request hangs,
   restart the tunnel first.
7. **Qwen3-Omni replies can be minutes long** — stream-play instead of
   waiting for completion, and size client timeouts generously.
