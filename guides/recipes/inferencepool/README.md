# InferencePool Recipe

This directory contains a standard `values.yaml` for deploying an `InferencePool`.

## Installation

To deploy the `InferencePool`, use the following Helm command, referencing the `values.yaml` in this directory.

```bash
helm install llm-d-infpool \
  -n ${NAMESPACE} \
  -f ./values.yaml \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool \
  --version v1.0.1
```

## Verification

You can verify the installation by checking the status of the created resources.

### Check the InferencePool

```bash
kubectl get inferencepool -n ${NAMESPACE}
```

```
NAME            AGE
llm-d-infpool   1m
```

### Check the Pods

```bash
kubectl get pods -l app.kubernetes.io/instance=llm-d-infpool -n ${NAMESPACE}
```

You should see the InferencePool's endpoint pod in a `Running` state.

```
NAME                                  READY   STATUS    RESTARTS   AGE
llm-d-infpool-epp-xxxxxxxx-xxxxx     1/1     Running   0          1m
```

