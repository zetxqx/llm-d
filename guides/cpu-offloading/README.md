# Well-lit Path: K-V Cache Offloading to CPU RAM

## Overview

This guide demonstrates how to deploy `meta-llama/Llama-3.3-70B-Instruct` using vLLM with CPU Offloading for KVCache.


## Hardware Requirements

This guide requires 12 Nvidia H100 GPUs. It requires 2400 Gi of memory across all 3 pods (800 Gi memory for each pod).

## Prerequisites

- Have the [proper client tools installed on your local system](../prereq/client-setup/README.md) to use this guide.
- Ensure your cluster infrastructure is sufficient to [deploy high scale inference](../prereq/infrastructure/README.md)
  - You must have high speed inter-accelerator networking
  - The pods leveraging inter-node EP must be deployed within the same networking domain
- Configure and deploy your [Gateway control plane](../prereq/gateway-provider/README.md).
- Have the [Monitoring stack](../../docs/monitoring/README.md) installed on your system.


## Installation

The Namespace in which the stack will be deployed will be derived from the `${NAMESPACE}` environment variable. If you have not set this, it will default to `llm-d-cpu-offloading` in this example.

```bash
export NAMESPACE=llm-d-cpu-offloading # or any other namespace
cd guides/cpu-offloading/
kubectl create namespace ${NAMESPACE}
```

### Create HuggingFace token

[Create the `llm-d-hf-token` secret in your ${NAMESPACE} with the key `HF_TOKEN` matching a valid HuggingFace token](../prereq/client-setup/README.md#huggingface-token) to pull models.

### Deploy Model Servers

GKE is tested Kubernetes providers for this well-lit path. You can customize the manifests if you run on other Kubernetes providers.

```bash
# Deploy on GKE
kubectl apply -k ./manifests/modelserver/gke -n ${NAMESPACE}
```

### Deploy InferencePool

```bash
# For GKE
helm install vllm-llama-3-70b-instruct \
  -n ${NAMESPACE} \
  --set inferencePool.modelServers.matchLabels.app=vllm-llama-3-70b-instruct \
  --set provider.name=gke \
  --set inferenceExtension.monitoring.gke.enabled=true \
  --version v1.0.1 \
  oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool

### Deploy Gateway and HTTPRoute

```bash
# Deploy a gke-l7-regional-external-managed gateway.
kubectl apply -k ./manifests/gateway/gke-l7-regional-external-managed -n ${NAMESPACE}
```

## Verifying the installation

- Firstly, you should be able to list all helm releases installed into your chosen namespace:

```bash
helm list -n ${NAMESPACE}
NAME                     	NAMESPACE           	REVISION	UPDATED                                	STATUS  	CHART               	APP VERSION
vllm-llama-3-70b-instruct	llm-d-cpu-offloading	1       	2025-10-05 21:00:01.334298353 +0000 UTC	deployed	inferencepool-v1.0.1	v1.0.1    
```

- Out of the box with this example you should have the following resources:

```bash
kubectl get all -n ${NAMESPACE}
NAME                                                READY   STATUS    RESTARTS   AGE
pod/vllm-llama-3-70b-instruct-84c57675f-4rdfv       1/1     Running   0          9m40s
pod/vllm-llama-3-70b-instruct-84c57675f-s4lps       1/1     Running   0          9m40s
pod/vllm-llama-3-70b-instruct-84c57675f-t5jxv       1/1     Running   0          9m40s
pod/vllm-llama-3-70b-instruct-epp-dff66b5f5-m4tg8   1/1     Running   0          6m23s

NAME                                             TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)             AGE
service/vllm-llama-3-70b-instruct-epp            ClusterIP   34.118.230.246   <none>        9002/TCP,9090/TCP   6m24s
service/vllm-llama-3-70b-instruct-ips-1ea1d340   ClusterIP   None             <none>        54321/TCP           6m23s

NAME                                            READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/vllm-llama-3-70b-instruct       3/3     3            3           9m41s
deployment.apps/vllm-llama-3-70b-instruct-epp   1/1     1            1           6m24s

NAME                                                      DESIRED   CURRENT   READY   AGE
replicaset.apps/vllm-llama-3-70b-instruct-84c57675f       3         3         3       9m41s
replicaset.apps/vllm-llama-3-70b-instruct-epp-dff66b5f5   1         1         1       6m24s
```

**_NOTE:_** This assumes no other guide deployments in your given `${NAMESPACE}` and you have not changed the default release names via the `${RELEASE_NAME}` environment variable.

## Using the stack

For instructions on getting started making inference requests see [our docs](../../docs/getting-started-inferencing.md)



## Cleanup

To remove the deployment:

```bash
# From examples/cpu-offloading-lws
helm uninstall vllm-llama-3-70b-instruct -n ${NAMESPACE}
kubectl delete -k ./manifests/modelserver/gke -n ${NAMESPACE}
kubectl delete -k ./manifests/gateway/gke-l7-regional-external-managed -n ${NAMESPACE}
```

## Customization

For information on customizing a guide and tips to build your own, see [our docs](../../docs/customizing-a-guide.md)
