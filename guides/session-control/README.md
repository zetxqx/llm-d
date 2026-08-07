# Session Control (Experimental)

## Overview

Minimum inference stack for the **Session Control Protocol experiment**
([llm-d/llm-d-router#2003](https://github.com/llm-d/llm-d-router/issues/2003)):
an aggregated SGLang fleet behind the llm-d router — 4 replicas (8×H100), so a wrong routing decision is observable (three "wrong" pods exist) and the KV-capacity knee lands inside the benchmark's concurrency sweep (~c64).

SGLang isolates a session's KV cache in a dedicated slot outside the shared
radix tree (`POST /open_session` / `POST /close_session`), exempt from LRU
while open and freed deterministically on close. Session state is
**per-worker**, which makes this a router problem: every turn of a session
must land on the pod holding its KV. This stack provides the substrate for
developing and benchmarking that router capability.

What routes **today** (all plugins exist in llm-d-router main):

- **[session-affinity-filter](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity)** — stateless pinning. The first response returns an
  `x-session-token` header (base64 pod name); the client echoes it on later
  turns and the filter narrows candidates to that pod.
- **[session-id-producer](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid)** — extracts `x-session-id` into the `SessionID`
  attribute. Consumed by nothing yet; it is the seam the proposed
  `session-binding-tracker` will read.
- **Load/prefix scorers** — first-turn placement and the fallback path.

What #2003 adds on top of this stack (see the design proposal): a server-side
session → pod binding table, lifecycle via `x-session-final` /
`/open_session` proxying, session-load-aware placement, and invalidation on
pod death. When those plugins land, only the plugins block in
[`router/session-control.values.yaml`](router/session-control.values.yaml)
changes — the deployment topology stays as-is.

## Default Configuration

| Parameter          | Value                                                                                                       |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Model              | [Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8)       |
| Engine             | SGLang                                                                                                      |
| Replicas           | 4                                                                                                           |
| Tensor Parallelism | 2                                                                                                           |
| GPUs per replica   | 2                                                                                                           |
| Total GPUs         | 8                                                                                                           |
| Context length     | 262,144 (native)                                                                                            |
| KV cache dtype     | fp8_e4m3                                                                                                    |

**Why this model.** The benchmark workload for this stack is agentic-trace replay (SemiAnalysis `cc-traces-weka-*`: median main-agent input ~195K tokens, 96% prefix reuse), which sets three hard requirements: ≥256K native context, standard GQA full attention (token-indexed KV, so radix/session-radix semantics and cache-hit metrics stay meaningful), and a small-activation MoE so 195K prefills are tractable. Qwen3-Coder-30B-A3B is the only current small model satisfying all three — the 2026 releases (Qwen3-Coder-Next, Qwen3.6, Gemma 4) all moved to hybrid linear/SWA attention. KV budget at this config: ~48KB/token FP8, ~9.4GB per median session, ~11–17 resident sessions per TP2 replica — the 4-replica fleet (~44–68 sessions) enters cache pressure around 64 concurrent sessions, which is the regime where affinity routing is measurable.

> [!NOTE]
> SGLang serves `/open_session` and `/close_session` unconditionally — no
> engine flag is required for dedicated-slot sessions. The session-tagged
> radix mode is enabled in
> [`modelserver/gpu/sglang/base/patch-sglang.yaml`](modelserver/gpu/sglang/base/patch-sglang.yaml)
> (`--enable-session-radix-cache`, which on this image requires
> `--radix-eviction-policy priority`); the base kustomization overrides the
> recipes component's image tag to v0.5.15.post1 where that mode first
> shipped. When upgrading past sglang#29173 (2026-08-02), session tracking
> moves to `UnifiedRadixCache`: add `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` and
> drop the priority-policy requirement.

## Prerequisites

- Have the [proper client tools installed on your local system](../../helpers/client-setup/README.md).
- Checkout the llm-d repo:

```bash
export branch="main" # branch, tag, or commit hash
git clone https://github.com/llm-d/llm-d.git && cd llm-d && git checkout ${branch}
```

- Set the following environment variables:

```bash
export REPO_ROOT=$(realpath $(git rev-parse --show-toplevel))
source ${REPO_ROOT}/guides/env.sh
export GUIDE_NAME="session-control"
export NAMESPACE="llm-d-${GUIDE_NAME}"
```

- Install the Gateway API Inference Extension CRDs:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml
```

- Create a target namespace for the installation:

```bash
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
```

## Installation Instructions

### 1. Prepare HF Token

Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 is public, but the secret makes swapping in a gated model a no-op. See [helpers/hf-token.md](../../helpers/hf-token.md).

```bash
export HF_TOKEN=<your HuggingFace token>
kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 2. Deploy the llm-d Router

Standalone mode (EPP with Envoy sidecar, no Kubernetes Gateway). The release
name `${GUIDE_NAME}` is mandatory — the inference pool selector matches the
guide label that pairs with this release.

```bash
helm install ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
  -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

For the benchmark's baseline arm, upgrade the same release to the llm-d optimized-baseline profile (approximate prefix-cache affinity, no session awareness — see [`router/optimized-baseline.values.yaml`](router/optimized-baseline.values.yaml), including its `peakPrefillThroughput` calibration note):

```bash
helm upgrade ${GUIDE_NAME} \
  ${ROUTER_STANDALONE_CHART} \
  -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
  -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/optimized-baseline.values.yaml \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}

# EPP reads the plugins ConfigMap at startup only — a mounted-ConfigMap update
# is NOT hot-reloaded, and if the pod template is unchanged helm won't roll the
# Deployment. Restart explicitly after every arm switch:
kubectl rollout restart deployment/${GUIDE_NAME}-epp -n ${NAMESPACE}
kubectl rollout status deployment/${GUIDE_NAME}-epp -n ${NAMESPACE} --timeout=5m
```

<details>
<summary><b>Gateway Mode</b></summary>

Deploy a Gateway named `llm-d-inference-gateway` per
[the gateway guides](../../docs/infrastructure/gateway), then:

```bash
export PROVIDER_NAME=istio   # options: none, gke, agentgateway, istio

helm install ${GUIDE_NAME} \
  ${ROUTER_GATEWAY_CHART} \
  -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
  -f ${REPO_ROOT}/guides/recipes/router/features/httproute-flags.yaml \
  -f ${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml \
  --set provider.name=${PROVIDER_NAME} \
  -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

> [!NOTE]
> For the explicit-lifecycle phase of #2003, the `HTTPRoute` must also match
> `/open_session` and `/close_session` so those calls reach the
> InferencePool.

</details>

### 3. Deploy the Model Server

```bash
export INFRA_PROVIDER=base # base | gke (gke adds an H100 nodeSelector)
kubectl apply -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/sglang/${INFRA_PROVIDER}/
```

Wait for the 4 decode pods to become ready (model download + engine warmup can take several minutes; 8 GPUs = 2 full H100 nodes, so the spot pool may need a scale-up first):

```bash
kubectl wait --for=condition=Ready pod \
  -l llm-d.ai/guide=${GUIDE_NAME} -n ${NAMESPACE} --timeout=30m
```

## Verification

### 1. Get the IP of the Proxy

```bash
export IP=$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```

<details>
<summary><b>Gateway Mode</b></summary>

```bash
export IP=$(kubectl get gateway llm-d-inference-gateway -n ${NAMESPACE} -o jsonpath='{.status.addresses[0].value}')
```

</details>

### 2. Send a Test Request

Open a temporary interactive shell inside the cluster:

```bash
kubectl run curl-debug --rm -it \
    --image=cfmanteiga/alpine-bash-curl-jq \
    --namespace="$NAMESPACE" \
    --env="IP=$IP" \
    -- /bin/bash
```

```bash
curl -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "prompt": "How are you today?"}' | jq
```

### 3. Verify Session Affinity

The first turn returns an `x-session-token` response header; decoding it
reveals the pod the request landed on. Echo the token on subsequent turns and
confirm every turn pins to that pod:

```bash
# Turn 1: capture the token (send x-session-id too — inert today, but it is
# the header the #2003 tracker will key on).
TOKEN=$(curl -si -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -H 'x-session-id: demo-session-1' \
    -d '{"model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8", "prompt": "Turn one."}' \
  | awk 'BEGIN{IGNORECASE=1} /^x-session-token:/ {print $2}' | tr -d '\r')
echo "pinned to: $(echo ${TOKEN} | base64 -d)"

# Turns 2..N: echo the token; the filter narrows candidates to the pinned pod.
for i in 2 3 4; do
  curl -si -X POST http://${IP}/v1/completions \
      -H 'Content-Type: application/json' \
      -H "x-session-id: demo-session-1" \
      -H "x-session-token: ${TOKEN}" \
      -d "{\"model\": \"Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8\", \"prompt\": \"Turn ${i}.\"}" \
    | awk 'BEGIN{IGNORECASE=1} /^x-session-token:/ {print $2}' | tr -d '\r' | base64 -d; echo
done
```

All turns should print the same pod name.

### 4. Exercise the SGLang Session API Directly

The router does not yet claim `/open_session` / `/close_session` — proxying
them with a tracked binding is precisely what #2003 adds. Until then,
exercise the engine API against a single pod:

```bash
kubectl port-forward -n ${NAMESPACE} deploy/session-control-gpu-sglang-decode 8000:8000 &

SESSION_ID=$(curl -s -X POST http://localhost:8000/open_session \
    -H 'Content-Type: application/json' \
    -d '{"capacity_of_str_len": 8192}' | tr -d '"')
echo "opened session: ${SESSION_ID}"

curl -s -X POST http://localhost:8000/close_session \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\": \"${SESSION_ID}\"}"
```

## Cleanup

```bash
helm uninstall ${GUIDE_NAME} -n ${NAMESPACE}
kubectl delete -n ${NAMESPACE} -k ${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/gpu/sglang/${INFRA_PROVIDER}/
```

## Scope Notes

- **Aggregated only** — P/D-disaggregated session placement is an explicit
  non-goal of the #2003 experiment; session KV pinning is a decode-pod
  concern and this stack deliberately has no prefill tier.
- **4 replicas (8 GPUs) is the benchmark configuration; 3 replicas is the experiment floor** if the spot pool cannot fit 2 full nodes. At ~11–17 resident 195K-sessions per TP2 replica (FP8 KV), the 8-GPU fleet saturates around c64 in weka-trace replay; scale `replicas` in [`patch-sglang.yaml`](modelserver/gpu/sglang/base/patch-sglang.yaml) when sweeping past c80 (c128 needs ~12 replicas or a HiCache tier).
- Stateless affinity trusts the client-echoed `x-session-token` and keeps no
  server-side state — no lifecycle, no capacity view, no invalidation. Those
  gaps are the point of the experiment.
