import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gateway Recipes

This directory contains recipes for deploying the `llm-d-inference-gateway` and `llm-d-route`.

## Installation

The following recipes are available for deploying the gateway and httproute.

<Tabs>
    <TabItem value="gke" label="GKE L7 Regional External Managed" default>
        This deploys a gateway suitable for GKE, using the `gke-l7-regional-external-managed` gateway class.

        ```bash
        kubectl apply -k ./gke-l7-regional-external-managed -n ${NAMESPACE}
        ```
    </TabItem>
    <TabItem value="istio" label="Istio">
        This deploys a gateway suitable for Istio, using the `istio` gateway class.

        ```bash
        kubectl apply -k ./istio -n ${NAMESPACE}
        ```
    </TabItem>
    <TabItem value="kgateway" label="KGateway">
        This deploys a gateway suitable for KGateway, using the `kgateway` gateway class.

        ```bash
        kubectl apply -k ./kgateway -n ${NAMESPACE}
        ```
    </TabItem>
    <TabItem value="kgateway-ocp" label="KGateway (OpenShift)">
        This deploys a gateway suitable for OpenShift, using the `openshift` gateway class.

        ```bash
        kubectl apply -k ./kgateway-openshift -n ${NAMESPACE}
        ```
    </TabItem>
</Tabs>

## Verification

You can verify the installation by checking the status of the created resources.

### Check the Gateway

```bash
kubectl get gateway -n ${NAMESPACE}
```

You should see output similar to the following, with the `PROGRAMMED` status as `True`. The `CLASS` will vary depending on the recipe you deployed.

```text
NAME                      CLASS                              ADDRESS     PROGRAMMED   AGE
llm-d-inference-gateway   gke-l7-regional-external-managed   <redacted>  True         1m
```

### Check the HTTPRoute

```bash
kubectl get httproute -n ${NAMESPACE}
```

```text
NAME          HOSTNAMES   AGE
llm-d-route               1m
```
