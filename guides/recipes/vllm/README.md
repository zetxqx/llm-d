# vLLM Recipe

This directory contains a standard recipe for deploying a vLLM model server.

## Installation

This is a base recipe and is meant to be used as a resource in other kustomizations. To deploy a standard vLLM model server, you can use the following command:

```bash
kubectl apply -k ./standard -n ${NAMESPACE}
```

Typically, you would overlay this base to provide specific arguments and resource requests for your model.
