# How the Dynamo container images are built

**Source:** `dynamo/` clone at commit `c914348c7` (main, 2026-07-03). All paths
relative to the dynamo repo root. Focus: the `vllm-runtime` image (the one that
serves vLLM and vLLM-Omni workers, and the one running in the `dynamo-system`
namespace of our cluster), with notes on the other images.

## TL;DR

There is **no static Dockerfile**. A Jinja2 renderer (`container/render.py`)
composes per-stage Dockerfile *templates* into one multi-stage Dockerfile,
parameterized by framework/device/target/arch, with all version pins coming
from `container/context.yaml`. Build =

```bash
container/render.py --framework vllm --target runtime --output-short-filename
docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
# (container/README.md:117-118)
```

## 1. Directory map (`container/`)

```
container/
├── render.py                  # Jinja2 renderer: framework × device × target × arch → Dockerfile
├── Dockerfile.template        # THE composition root: {% include %}s stage templates
├── context.yaml               # single source of version pins (per-framework sections)
├── templates/                 # one Jinja template per build stage / image flavor
│   ├── args.Dockerfile        #   global ARGs (BASE_IMAGE, VLLM_OMNI_REF, …) from context.yaml
│   ├── dynamo_base.Dockerfile #   tools stage: rust, uv, sccache, nats-server, etcd
│   ├── wheel_builder.Dockerfile #  manylinux stage: builds all Dynamo wheels + NIXL/UCX
│   ├── vllm_runtime.Dockerfile  #  final vLLM/vLLM-Omni runtime image
│   ├── sglang_runtime.Dockerfile / trtllm_runtime.Dockerfile   # sibling frameworks
│   ├── dynamo_runtime.Dockerfile / frontend.Dockerfile / planner.Dockerfile
│   ├── dev.Dockerfile / local_dev.Dockerfile   # dev targets (source-mounted workflows)
│   ├── aws.Dockerfile (EFA) / compliance.Dockerfile / sglang_xpu_framework.Dockerfile
├── deps/
│   ├── requirements.{common,vllm,sglang,trtllm,frontend,planner,dev,test,benchmark}.txt
│   └── vllm/
│       ├── install_vllm_omni.sh      # vllm-omni wheel install + site-packages patch
│       └── protected_packages.txt    # packages whose versions must not move
├── compliance/                # SBOM auditing, license harvesting (NOTICES) tooling
├── launch_message/            # banners shown at container start
├── run.sh                     # convenience wrapper to run built images (GPU flags, mounts)
└── use-sccache.sh             # sccache (S3-backed Rust/C++ build cache) helper
```

## 2. The render pipeline

`container/render.py`:
- Inputs: `--framework {dynamo,vllm,sglang,trtllm}` × `--device {cuda,xpu,cpu}`
  × `--target {runtime,dev,local-dev,wheel_builder,base,frontend,planner}` ×
  `--platform` (amd64/arm64/multi) × `--cuda-version {13.0,13.1}` (13.1 =
  trtllm only). A validation matrix (`validate_args`) rejects unsupported
  combos.
- Loads `context.yaml` into the Jinja context, renders `Dockerfile.template`
  with `StrictUndefined` (missing pins fail the render, not the build).
- Output: `<framework>-<target>-cuda<ver>-<arch>-rendered.Dockerfile` (or
  `rendered.Dockerfile` with `--output-short-filename`).

`Dockerfile.template` is pure composition — for `--framework vllm` it includes,
in order: `args` → `dynamo_base` → `wheel_builder` → `vllm_runtime` (plus
`aws`/`dev`/`local_dev` conditionally). Framework `dynamo` swaps the last for
`dynamo_runtime`/`frontend`/`planner` per target.

`context.yaml` has one section per framework. Key pins in the `vllm:` section:
upstream `runtime_image: vllm/vllm-openai` (the TRUE base of the final image),
`vllm_omni_ref: "v0.23.0rc1"` (line 80), `flashinf_ref: v0.6.8.post1`, plus
compliance baselines (`baseline_sbom`). `templates/args.Dockerfile` turns these
into `ARG`s (e.g. line 103: `ARG VLLM_OMNI_REF={{ context.vllm.vllm_omni_ref }}`).

## 3. Stage graph (vllm / cuda / runtime)

```
quay.io/pypa/manylinux_2_28_{x86_64,aarch64}      ${BASE_IMAGE} (nvidia/cuda…)
                 │                                        │
                 ▼                                        ▼
        [wheel_builder_base]                        [dynamo_base]
          system deps, CUDA copy,                     rust 1.93.1, uv, sccache,
          gcc-14 toolset, protoc,                     nats-server (ARG NATS_VERSION),
          maturin/meson/pybind11                      etcd (ARG ETCD_VERSION)
                 │                                        │
                 ▼                                        │
        [runtime_wheel_builder]                           │
          COPY Cargo.toml, lib/, components/…             │
          → uv build            = ai_dynamo-1.3.0 (pure py wheel, root pyproject.toml)
          → maturin build       = ai_dynamo_runtime (Rust ext, lib/bindings/python)
              --features kv-indexer,slot-tracker,select-service,mm-routing,
                         aic-forward-pass[,media-ffmpeg]
          → uv build lib/gpu_memory_service = gpu_memory_service wheel
          → meson: NIXL (cloned from ai-dynamo/nixl) + UCX  → nixl wheel
                 │                                        │
                 └───────────────┬────────────────────────┘
                                 ▼
                    [pre_runtime → runtime]  FROM vllm/vllm-openai:<tag>
                                 │
                                 ▼
                nvcr.io/nvidia/ai-dynamo/vllm-runtime:<ver>
```

(Stage names and structure: `wheel_builder.Dockerfile:15-27,494-527,605-679`;
`vllm_runtime.Dockerfile:11-15`; sccache/S3 caching for the Rust builds via
`use-sccache.sh`, cargo/uv layer caches via BuildKit cache mounts.)

## 4. What each stage produces

### `dynamo_base` (`templates/dynamo_base.Dockerfile`)
Tool depot, later COPY'd from — not a parent of the final image: sccache
binary, `uv`/`uvx` (from `ghcr.io/astral-sh/uv:latest` — noted TODO: unpinned),
**nats-server** (deb), **etcd** (tarball → `/usr/local/bin/etcd`), Rust
toolchain **1.93.1** via rustup.

### `wheel_builder` (`templates/wheel_builder.Dockerfile`)
Runs on **manylinux_2_28** (not the CUDA image) so wheels are portable.
Produces into `/opt/dynamo/dist/`:

| Wheel | Built from | How |
| --- | --- | --- |
| `ai_dynamo-1.3.0-*any.whl` | root `pyproject.toml` (+`hatch_build.py`), includes `components/` Python | `uv build --wheel` (line 522) |
| `ai_dynamo_runtime-*.whl` | `lib/bindings/python` (PyO3/maturin) — the Rust `DistributedRuntime` bindings | `maturin build --release --features …mm-routing…` (lines 525-527) |
| `gpu_memory_service-*.whl` | `lib/gpu_memory_service` (small C++ ext) | `uv build` (line 613) |
| `nixl-*.whl` | `git clone https://github.com/ai-dynamo/nixl.git`, meson vs `/opt/nvidia/nvda_nixl` (or Intel prefix for xpu) | lines 647-679 |

Also: UCX build (with an Intel patch for xpu, line 78), license harvesting from
the cargo registry for NOTICES (compliance).

### `vllm_runtime` (`templates/vllm_runtime.Dockerfile`) — the final image
- **Base:** `FROM ${RUNTIME_IMAGE}` = **`vllm/vllm-openai`** per arch (lines
  11-15). The upstream vLLM/torch/CUDA python env is inherited as-is.
- **COPY in from earlier stages:** nats-server, etcd, uv (71-73, from
  `dynamo_base`); UCX (91) and the nixl wheel (99) from `wheel_builder`.
- **Non-root user:** creates `dynamo` user, chowns `/workspace`, `/opt/dynamo`
  (84).
- **pip installs (all `--no-deps`, order matters):** nixl wheel (142) →
  `ai_dynamo_runtime` (154) → `ai_dynamo` (155) → optional KVBM/GMS wheels
  (158-162) → framework requirements from `deps/requirements.vllm.txt`.
- **vLLM-Omni install** (181-187): mounts `deps/vllm/protected_packages.txt` +
  `install_vllm_omni.sh`, which (a) freezes already-installed packages into a
  constraints file, (b) `uv pip install vllm-omni==${VLLM_OMNI_REF#v}`
  (`v0.23.0rc1` → `0.23.0rc1`, `--prerelease=allow`), (c) **applies a curl'd
  git patch onto site-packages** (vllm-omni commit `17cf60a`, fixing an
  `OmniRequest.__init__` bug that broke every vLLM worker because vllm-omni
  monkeypatches `vllm.v1.request.Request` at import time).
- **Source trees into `/workspace`** (283-289): `tests`, `examples`, `dev`,
  `components/src/dynamo/{common,frontend,vllm}`, `lib` — this is why
  DynamoGraphDeployments can set `workingDir: /workspace/examples/backends/vllm`
  and why the omni glue is present both as installed wheel *and* as browsable
  source.
- **Cleanup:** `rm -rf /workspace/vllm` (276) — upstream vLLM source is purged;
  only its installed site-packages remain. GPL media codecs from the upstream
  image are purged for license compliance (context.yaml comments), with an
  LGPL ffmpeg CLI kept because omni examples need it (165-167, 248).

## 5. Targets and sibling images

Same machinery, different composition (`render.py validate_args` +
`Dockerfile.template` branches):

| Image | Framework/target | Final base | Contents |
| --- | --- | --- | --- |
| `vllm-runtime` | vllm/runtime | `vllm/vllm-openai` | above |
| `sglang-runtime` | sglang/runtime | `lmsysorg/sglang` | analogous (`sglang_runtime.Dockerfile`) |
| `tensorrtllm-runtime` | trtllm/runtime (CUDA 13.1) | `nvcr.io/nvidia/tensorrt-llm/release` | analogous |
| `dynamo frontend` | dynamo/frontend | slim | Rust frontend + router only |
| `dynamo-planner` | dynamo/planner | slim | SLA planner |
| dev / local-dev | any/dev | framework image | + dev deps, source-mount conveniences (`dev.Dockerfile`) |

Published to NGC as `nvcr.io/nvidia/ai-dynamo/<name>:<release>` (1.2.1 / 1.3.0
tags current in docs; our cluster runs 1.0.2). The k8s operator's
DynamoGraphDeployment services reference these images and select the role via
`command:` (`python -m dynamo.vllm`, `python -m dynamo.vllm.omni`, …) — see
[dynamo-diffusion-omni-report.md](dynamo-diffusion-omni-report.md) Step 1.

## 6. Quirks worth remembering

1. **The Dockerfile you build is generated** — never edit a rendered file;
   change `templates/*` or `context.yaml` and re-render.
2. **Wheels are built on manylinux, not on the CUDA base** — Rust/C++ artifacts
   are portable and cached (sccache→S3, cargo/uv BuildKit mounts); the final
   image never contains a compiler.
3. **Dynamo code enters the image twice**: as installed wheels
   (`ai_dynamo`, `ai_dynamo_runtime`) *and* as `/workspace` source for
   examples/tests/workingDir. The wheel is what `python -m dynamo.vllm.omni`
   imports.
4. **vLLM itself is never built** — inherited from the official upstream image;
   Dynamo pins it only indirectly via the image tag in `context.yaml`.
5. **vLLM-Omni is a pinned prerelease wheel patched in-place at build time** —
   the tightest and most fragile coupling in the whole build; watch
   `install_vllm_omni.sh` when comparing versions (our stacks run v0.22.0,
   this image ships 0.23.0rc1).
6. **`mm-routing` is a compile-time Rust feature** of `ai_dynamo_runtime`
   (wheel_builder line 525) — multimodal-hash routing support is baked into
   the wheel, not a runtime plugin.
