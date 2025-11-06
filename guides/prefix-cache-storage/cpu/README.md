# Offloading Prefix Cache to CPU Memory

## Overview

This guide provides recipes to offload prefix cache to CPU RAM via the vLLM native offloading connector and the LMCache connector.

## Prerequisites

* All prerequisites from the [upper level](../README.md).
* TODO: Add other prerequisites.

## Installation

### Deploy vLLM

=== vLLM Native Offloading [To be added]
This enables CPU prefix cache offloading via the native vLLM OffloadingConnector.

```
kubectl apply -k ./manifests/vllm/offloading-connector
```

=== Via LMCache [To be added]
This enables CPU prefix cache offloading via the LMCacheConnector.

```
kubectl apply -k ./manifests/vllm/lmcache-connector
```

=== TPU to CPU Offloading [To be added]
This enables CPU prefix cache offloading for vLLM on TPU.

```
kubectl apply -k ./manifests/vllm/tpu
```

### [TODO] Deploy other resources
