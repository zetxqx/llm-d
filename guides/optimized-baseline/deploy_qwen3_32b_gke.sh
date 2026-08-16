#!/usr/bin/env bash
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

# 1. Determine REPO_ROOT dynamically
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

echo "Setting REPO_ROOT to: ${REPO_ROOT}"

# 2. Check dependencies
for cmd in kubectl helm git; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "Error: $cmd is required but not installed." >&2
    exit 1
  fi
done

# 3. Source environment variables
ENV_SH="${REPO_ROOT}/guides/env.sh"
if [[ -f "${ENV_SH}" ]]; then
  echo "Sourcing environment variables from ${ENV_SH}..."
  source "${ENV_SH}"
else
  echo "Error: ${ENV_SH} not found. Are you running this from the llm-d repository?" >&2
  exit 1
fi

# 4. Check kubectl cluster connection
echo "Checking connection to Kubernetes cluster..."
if ! kubectl cluster-info &> /dev/null; then
  echo "Error: Cannot connect to Kubernetes cluster. Please verify your kubeconfig and GKE context." >&2
  exit 1
fi

# 5. Get or prompt HuggingFace Token
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HuggingFace Token (HF_TOKEN) is not set."
  read -rsp "Please enter your HuggingFace Token: " HF_TOKEN
  echo ""
  if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN cannot be empty." >&2
    exit 1
  fi
  export HF_TOKEN
fi

# Set deploy variables
export GUIDE_NAME="optimized-baseline"
export NAMESPACE="llm-d-optimized-baseline"
export ACCELERATOR_TYPE="gpu"
export MODEL_SERVER="vllm"
export INFRA_PROVIDER="gke"

echo "=== Deployment Configuration ==="
echo "Namespace:        ${NAMESPACE}"
echo "Guide Name:       ${GUIDE_NAME}"
echo "Accelerator Type: ${ACCELERATOR_TYPE}"
echo "Model Server:     ${MODEL_SERVER}"
echo "Infra Provider:   ${INFRA_PROVIDER}"
echo "GAIE Version:     ${GAIE_VERSION}"
echo "Router Version:   ${ROUTER_CHART_VERSION}"
echo "================================"

# 6. Install Gateway API Inference Extension CRDs
echo "Installing Gateway API Inference Extension CRDs (version ${GAIE_VERSION})...."
kubectl apply -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${GAIE_VERSION}/v1-manifests.yaml"

# 7. Create Namespace
echo "Creating namespace ${NAMESPACE}..."
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# 8. Create HuggingFace Secret
echo "Creating HuggingFace token secret..."
kubectl create secret generic llm-d-hf-token \
  --from-literal="HF_TOKEN=${HF_TOKEN}" \
  --namespace "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

# 9. Deploy the llm-d Router (Standalone Mode)
echo "Deploying llm-d Router in Standalone Mode..."
helm upgrade --install "${GUIDE_NAME}" \
    "${ROUTER_STANDALONE_CHART}" \
    -f "${REPO_ROOT}/guides/recipes/router/base.values.yaml" \
    -f "${REPO_ROOT}/guides/${GUIDE_NAME}/router/${GUIDE_NAME}.values.yaml" \
    --namespace "${NAMESPACE}" \
    --version "${ROUTER_CHART_VERSION}"

# 10. Prepare customization for Model Server (3 replicas, H100 node selector)
CUSTOM_PATCH_DIR="${REPO_ROOT}/guides/${GUIDE_NAME}/modelserver/${ACCELERATOR_TYPE}/${MODEL_SERVER}/${INFRA_PROVIDER}"
echo "Writing customization patches to ${CUSTOM_PATCH_DIR}..."

cat <<EOF > "${CUSTOM_PATCH_DIR}/patch-custom.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: decode
spec:
  replicas: 3
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-h100-80gb
EOF

cat <<EOF > "${CUSTOM_PATCH_DIR}/kustomization.yaml"
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../base
components:
  - ../../../../../recipes/modelserver/components/disable-gke-nccl-tuner-patch
patches:
  - path: patch-custom.yaml
EOF

# 11. Deploy the Model Server
echo "Deploying Model Server (Qwen3-32B vLLM)..."
kubectl apply -n "${NAMESPACE}" -k "${CUSTOM_PATCH_DIR}/"

echo ""
echo "========================================================================="
echo "Deployment initiated successfully!"
echo "========================================================================="
echo ""
echo "To monitor the deployment, run:"
echo "  kubectl get pods -n ${NAMESPACE} -w"
echo ""
echo "Once the model servers are running (it may take several minutes to download the model),"
echo "verify the stack with the following commands:"
echo ""
echo "1. Get the Proxy IP:"
echo "   export IP=\$(kubectl get service ${GUIDE_NAME}-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')"
echo ""
echo "2. Run a temporary interactive debug pod inside the cluster:"
echo "   kubectl run curl-debug --rm -it \\"
echo "       --image=cfmanteiga/alpine-bash-curl-jq \\"
echo "       --namespace=\"\${NAMESPACE}\" \\"
echo "       --env=\"IP=\${IP}\" \\"
echo "       --env=\"NAMESPACE=\${NAMESPACE}\" \\"
echo "       -- /bin/bash"
echo ""
echo "3. Inside the debug pod, run the test curl command:"
echo "   curl -X POST http://\${IP}/v1/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{"
echo "           \"model\": \"Qwen/Qwen3-32B\","
echo "           \"prompt\": \"How are you today?\""
echo "       }' | jq"
echo "========================================================================="
