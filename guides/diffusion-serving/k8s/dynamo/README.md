# Minimal vLLM-Omni text-to-image deployment on Kubernetes

Deploys one Dynamo frontend and one vLLM-Omni worker serving `Qwen/Qwen-Image`
(text-to-image) as a `DynamoGraphDeployment`. This is the k8s equivalent of
`examples/backends/vllm/launch/agg_omni_image.sh` in the dynamo repo.

## Prerequisites

1. A cluster with at least one GPU node big enough for the model
   (`Qwen/Qwen-Image` is ~20B parameters; use an 80GB-class GPU, or pass a
   smaller model via `--model`).
2. The Dynamo platform (operator + etcd + NATS) installed, e.g.:

   ```bash
   export NAMESPACE=dynamo-system
   helm install dynamo-platform oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
     --namespace $NAMESPACE --create-namespace
   ```

   See the dynamo repo's `docs/kubernetes/installation_guide.md` for details.
3. A HuggingFace token secret in the same namespace:

   ```bash
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN=<your-token> -n $NAMESPACE
   ```

4. Edit `agg_omni_image.yaml` and replace `my-tag` (both occurrences) with a
   real `vllm-runtime` tag from NGC. vLLM-Omni is pre-installed in the
   container images (amd64 only), so no extra install step is needed.

## Deploy

```bash
kubectl apply -f agg_omni_image.yaml -n $NAMESPACE
kubectl get dynamographdeployment -n $NAMESPACE
kubectl get pods -n $NAMESPACE   # wait for frontend + worker to be Running
```

First startup is slow: the worker downloads the model weights before it
registers with the frontend.

## Test it

```bash
kubectl port-forward svc/vllm-omni-image-frontend 8000:8000 -n $NAMESPACE
```

Image generation endpoint (ask for `b64_json` so the image comes back inline —
the default `url` format points at the worker pod's local filesystem):

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "A red apple on a white table",
    "size": "512x512",
    "num_inference_steps": 20,
    "response_format": "b64_json"
  }' | jq -r '.data[0].b64_json' | base64 -d > apple.png
```

Or via chat completions, which returns base64 images inline by default:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen-Image",
    "messages": [{"role": "user", "content": "A cat sitting on a windowsill"}],
    "stream": false
  }'
```

## Cleanup

```bash
kubectl delete -f agg_omni_image.yaml -n $NAMESPACE
```
