# Deploying Adam Platform to Kubernetes

This guide explains how to deploy the Adam Platform using the provided Kubernetes (IaC) manifests.

## Prerequisites

*   A running Kubernetes cluster (e.g., Minikube, EKS, GKE).
*   `kubectl` configured to communicate with your cluster.

## Deployment Steps

1.  **Apply the OS Kernel Manifest:**
    This deploys the deterministic execution layer and persistent storage.
    ```bash
    kubectl apply -f kubernetes/adam-os-kernel.yaml
    ```

2.  **Apply the Sidecar Agents Manifest:**
    This deploys the probabilistic multi-agent swarm.
    ```bash
    kubectl apply -f kubernetes/sidecar-agents.yaml
    ```

3.  **Verify Deployment:**
    Ensure all pods are in the `Running` state.
    ```bash
    kubectl get pods -l 'app in (adam-os, adam-sidecar)'
    ```
