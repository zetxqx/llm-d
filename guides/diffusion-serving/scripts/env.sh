# Shared environment for the diffusion-serving guide scripts. Source this first:
#   source scripts/env.sh
# The guide is self-contained in the llm-d repo; the only external checkouts
# are opt-in via env vars:
#   VLLM_OMNI_DIR — vllm-omni checkout, source of the benchmark client
#                   (required by calibrate.sh and run_arm.sh)
#   SCHED_DIR     — llm-d-inference-scheduler checkout with the diffusion cost
#                   plugins (required by build_epp_image.sh only)

# Resolve the guide folder and the llm-d repo root from this file's location.
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export BENCH_DIR="$(dirname "$_ENV_SH_DIR")"
export LLM_D_DIR="$(cd "$BENCH_DIR/../.." && pwd)"

export NAMESPACE="${NAMESPACE:-llm-d-omni}"

# Helm chart coordinates (ROUTER_STANDALONE_CHART, ROUTER_CHART_VERSION, ...).
source "$LLM_D_DIR/guides/env.sh"

# EPP image: MUST be built from llm-d-inference-scheduler branch
# feat/diffusion-declared-cost (see scripts/build_epp_image.sh) — the older
# images-gen-v2 tag predates the diffusion-cost-scorer.
export EPP_REGISTRY="${EPP_REGISTRY:-us-central1-docker.pkg.dev/bobzetian-gke-dev/bobinference}"
export EPP_TAG="${EPP_TAG:-declared-cost-v1}"

# Model pool size. Keep in sync with manifests/modelserver/kustomization.yaml.
export REPLICAS="${REPLICAS:-3}"

# llm-d pool identity (matches the k8s/ overlays).
export HELM_RELEASE="llm-d-omni-qwen-image"
export POOL_SELECTOR="llm-d.ai/guide=optimized-baseline-omni-qwen-image"
export EPP_DEPLOY="llm-d-omni-qwen-image-epp"
export EPP_SVC_URL="http://llm-d-omni-qwen-image-epp.${NAMESPACE}.svc.cluster.local:80"
export BASELINE_SVC_URL="http://qwen-baseline.${NAMESPACE}.svc.cluster.local:8000"

# Measured 2-pod capacity in req/s; overwritten by scripts/calibrate.sh into
# results/capacity.env, which run_arm.sh sources when present.
export CAPACITY_RPS="${CAPACITY_RPS:-0.5}"
