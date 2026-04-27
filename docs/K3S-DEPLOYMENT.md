# K3s Deployment Guide

How to run the speech container as a K3s pod instead of standalone Docker.

## Why K3s Over Docker?

| Feature | Docker | K3s |
|---------|--------|-----|
| Auto-restart on crash | `--restart` flag | ✅ Built-in (Deployment) |
| Rolling updates | Manual | ✅ Native |
| Resource limits | `--memory` flag | ✅ Declarative YAML |
| Health checks | HEALTHCHECK only | ✅ liveness + readiness |
| Arc visibility | ❌ Invisible | ✅ Arc-managed workload |
| GPU scheduling | `--gpus all` | ✅ `nvidia.com/gpu: 1` |
| Multi-container | Compose | ✅ Native pods/sidecars |

## Prerequisites

1. GPU Operator or NVIDIA device plugin installed
2. `nvidia` RuntimeClass created
3. Local registry running at `localhost:5000`
4. `/acsa/hf-models` directory created

## Deploy

```bash
# 1. Push image to local registry
docker tag mic-access:latest localhost:5000/mic-access:latest
docker push localhost:5000/mic-access:latest

# 2. Deploy
sudo k3s kubectl apply -f k8s/speech-recognizer.yaml

# 3. Watch
sudo k3s kubectl -n speech get pods -w
sudo k3s kubectl -n speech logs -f deployment/speech-recognizer
```

## Architecture

```
K3s Cluster
├── speech namespace
│   └── speech-recognizer (Deployment, 1 replica)
│       ├── hostNetwork: true (for /dev/snd)
│       ├── nvidia.com/gpu: 1
│       ├── memory: 8Gi limit
│       ├── cpus: 4 limit
│       └── PVC → /acsa/hf-models
├── foundry-local-operator namespace (future)
│   └── inference-operator
├── cert-manager namespace
│   ├── cert-manager
│   ├── cert-manager-cainjector
│   ├── cert-manager-webhook
│   └── trust-manager
└── azure-arc namespace (management)
```

## GPU Setup for K3s

```bash
# Install GPU Operator (driver already on host)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --set driver.enabled=false \
  --set toolkit.enabled=false \
  --set operator.defaultRuntime=containerd

# Create RuntimeClass
kubectl apply -f - <<EOF
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

# Verify GPU visible to K3s
kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
# Expected: "1"
```

## Cleanup

```bash
sudo k3s kubectl delete -f k8s/speech-recognizer.yaml
```
