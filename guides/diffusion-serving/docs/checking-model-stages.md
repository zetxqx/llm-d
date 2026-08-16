# How to check a model's stages in vLLM-Omni (worked examples: Z-Image, Qwen3-TTS)

A practical guide: given a model name, how to find out whether vLLM-Omni
supports it, how many stages it runs, what *type* each stage is
(AR / one-shot generation / diffusion), and where the code and config live.
All paths relative to the `vllm-omni/` clone.

## The four places to look

| Question | Where to look |
| --- | --- |
| Is the model supported? | `docs/models/supported_models.md` (the official table; architecture names ending in `*Pipeline` = diffusion, `*ForConditionalGeneration`/`*Model` = AR/omni) |
| How many stages, on which GPUs, with what knobs? | `vllm_omni/deploy/<model>.yaml` — the `stages:` list. **No yaml = single-stage diffusion** (config is synthesized). |
| What TYPE is each stage? | The model's frozen topology: `vllm_omni/model_executor/models/<model>/pipeline.py` → `StagePipelineConfig.execution_type` |
| Where is the model code? | AR/omni: `vllm_omni/model_executor/models/<model>/`; diffusion: `vllm_omni/diffusion/models/<model>/` (registered in `vllm_omni/diffusion/registry.py`) |

**Key design fact:** the deploy yaml carries *tuning only* (devices, memory,
batch sizes, sampling defaults). The topology — stage count semantics, stage
types, dataflow — is **frozen in code** (`stage_config.py:203-205`:
"Fixed topology for one stage (frozen, not user-configurable)"). Yaml
comments like `# Stage 0: AR Model` are documentation, not configuration.
The two join on `stage_id`.

## The stage-type vocabulary

`StageExecutionType` (`vllm_omni/config/stage_config.py:167-173`) — three
values, each selecting a different scheduler (`:185-192`):

| `execution_type` | Meaning | Scheduler |
| --- | --- | --- |
| `LLM_AR` | autoregressive token loop; KV cache; continuous batching | `OmniARScheduler` (sync or async) |
| `LLM_GENERATION` | one-shot parallel forward, no AR loop (vocoders) | `OmniGenerationScheduler` |
| `DIFFUSION` | denoising-loop engine | diffusion engine's own FIFO scheduler |

Other `StagePipelineConfig` fields worth reading: `model_stage` (human role
name), `input_sources` (dataflow edges between stages), `final_output` +
`final_output_type` (which modality this stage can terminate — this is what
the request's `modalities` field is matched against, see
[vllm-omni-stage-pipeline.md](vllm-omni-stage-pipeline.md) §4), and
`engine_output_type` (what it emits: `token_ids`, `latent`, `audio`, …).

**Yaml-only heuristic:** if you only have a deploy yaml, the
`default_sampling_params` betray the stage type — `temperature/top_k/
stop_token_ids/max_tokens` = AR; `num_inference_steps/guidance_scale/
height/width` = diffusion.

---

## Worked example 1: Z-Image-Turbo (single-stage diffusion)

**The trail:**

1. **Supported?** `docs/models/supported_models.md`: row `ZImagePipeline` /
   `Tongyi-MAI/Z-Image-Turbo`. `*Pipeline` name → diffusion family.
2. **Deploy yaml?** `ls vllm_omni/deploy/ | grep -i z_image` → **nothing**.
   That absence *is* the answer: single-stage diffusion models need no yaml.
   At startup `ConfigFactory.create_default_diffusion()`
   (`vllm_omni/config/config_factory.py:200-240` — docstring: "Single-stage
   diffusion — no YAML needed") synthesizes one stage in memory:
   `model_stage="diffusion"`, devices from the parallel config,
   `cache_backend: "none"`. Per-stage knobs come from CLI flags instead
   (`--enable-cpu-offload`, `--vae-use-slicing`, …).
3. **How it's recognized:** the HF checkpoint is diffusers-format — no
   `model_type`, but a `model_index.json` naming the class. Resolution:
   `model_index.json` → `ZImagePipeline` → diffusion registry
   (`vllm_omni/diffusion/registry.py:49-52`) → module `z_image`.
4. **Model code:** `vllm_omni/diffusion/models/z_image/`
   (`pipeline_z_image.py` — encode/denoise/decode; `z_image_transformer.py`
   — the DiT). Weights layout in the checkpoint mirrors the components:
   `text_encoder/`, `transformer/`, `vae/`, `scheduler/` subfolders.

**Result:** 1 stage, type DIFFUSION, containing text-encode + DiT loop + VAE
decode internally. (RFC #4590 proposes splitting those into 3 stages — when
that lands, Z-Image-class models would gain a real deploy yaml.)

**Stage map:**

```
stage 0  [DIFFUSION]  text_encoder → DiT ×N steps → VAE  → image (final)
```

---

## Worked example 2: Qwen3-TTS (two-stage AR → vocoder)

**The trail:**

1. **Supported?** In the supported-models table; architecture
   `Qwen3TTSTalkerForConditionalGeneration` (AR-family name).
2. **Deploy yaml:** `vllm_omni/deploy/qwen3_tts.yaml` — 2 entries in
   `stages:` (plus tuning variants `qwen3_tts_high_concurrency.yaml`,
   `qwen3_tts_forced_aligner.yaml` — same topology, different knobs).
   Header comment: "talker → code2wav via shared-memory chunk streaming".
3. **Stage types:** `vllm_omni/model_executor/models/qwen3_tts/pipeline.py`:

   ```python
   StagePipelineConfig(stage_id=0, model_stage="qwen3_tts",
       execution_type=StageExecutionType.LLM_AR,        # AR talker
       engine_output_type="latent", owns_tokenizer=True, ...)
   StagePipelineConfig(stage_id=1, model_stage="code2wav",
       execution_type=StageExecutionType.LLM_GENERATION, # one-shot vocoder
       input_sources=(0,),
       final_output=True, final_output_type="audio", ...)
   ```

4. **Model code:** `vllm_omni/model_executor/models/qwen3_tts/`; registered
   in `model_executor/models/registry.py` (`_OMNI_MODELS`).

**Result:** 2 stages — an `LLM_AR` talker (text → RVQ codec tokens,
LLM-style) and an `LLM_GENERATION` vocoder (codec → waveform, single
parallel forward, **not** diffusion and not AR). Connected by a
SharedMemoryConnector with chunk streaming (`initial_codec_chunk_frames: 1`
— the very first audio chunk ships after ONE codec frame; that's the ~13 ms
TTFB we measured through the llm-d router).

**Stage map:**

```
stage 0  [LLM_AR]          text → codec tokens (KV cache, continuous batching)
   │  SHM connector, chunk streaming (first chunk = 1 frame)
   ▼
stage 1  [LLM_GENERATION]  codec tokens → waveform  → audio (final)
```

**Notable yaml findings while reading `qwen3_tts.yaml`:**

- **`enable_prefix_caching: true` on the talker stage** (with a comment:
  "any text repetition across requests (same voice or shared ref_audio) is
  a net throughput win"). Unlike the omni pipelines (Qwen3-Omni disables it
  on every stage pending RFC #1184), TTS's talker ships with prefix caching
  ON in current main — so **prefix-cache-affinity routing for the TTS stack
  has live signal today**, not just post-RFC.
- Per-stage asymmetry in one glance: talker `max_num_batched_tokens: 512`
  (latency tuning: small per-step batches keep first-chunk latency low) vs
  code2wav `65536` (correctness: codec prefill length exceeds the 32k
  default). Same file, two stages, two entirely different tuning regimes —
  the per-stage-heterogeneity argument in miniature.

---

## Quick recipe (any model)

```bash
cd vllm-omni
# 1. supported?
grep -i "<model>" docs/models/supported_models.md
# 2. deploy yaml → stage count (count above the platforms: section!)
ls vllm_omni/deploy/ | grep -i "<model>"
awk '/^platforms:/{exit} /stage_id:/{c++} END{print c" stages"}' vllm_omni/deploy/<model>.yaml
# 3. stage types (authoritative)
cat vllm_omni/model_executor/models/<model>/pipeline.py   # look for execution_type
# (older models: vllm_omni/model_executor/stage_configs/<model>.yaml)
# 4. no deploy yaml + listed as *Pipeline?  → single-stage diffusion
```

Caveat on step 2: the `platforms:` section repeats `stage_id` entries as
per-hardware deltas — `grep -c stage_id` overcounts (qwen3_omni_moe.yaml
grep-counts 10 but has 3 real stages).

## Why this matters for routing

The topology is **static metadata per model** — stage count, each stage's
execution type, and which stage terminates which modality are knowable
without probing any running pod. A stage-pool router (Tier 4 in
[routing-opportunities.md](routing-opportunities.md)) can load the same
`PipelineConfig` the engine trusts and immediately know: Qwen3-TTS = an
AR pool (cache-affinity + queue-depth signals, prefix caching live today)
feeding a vocoder pool (one-shot, throughput-bound); Z-Image = a single
diffusion pool (declared-cost signal, no cache). The routing policy per pool
falls out of `execution_type`.
