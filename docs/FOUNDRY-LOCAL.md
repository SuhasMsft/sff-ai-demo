# Foundry Local — Analysis & Future Use

## TL;DR

**Foundry Local is NOT suitable for NeMo Parakeet TDT 0.6B** (our speech model). It's designed for text LLMs (Phi-4, Qwen) with ONNX runtime. Keep direct NeMo loading for ASR.

## Why It Doesn't Work for Our Model

| Requirement | Foundry Local | Parakeet TDT 0.6B |
|-------------|---------------|-------------------|
| Model format | ONNX only | `.nemo` (PyTorch) |
| API pattern | `/v1/chat/completions` or `/v1/predict` | Audio → text (ASR pipeline) |
| Preprocessing | None (raw tensors) | Mel spectrogram + feature normalization |
| Decoding | None (raw output) | TDT beam search (iterative, stateful) |
| Architecture | Single ONNX graph | Encoder + Predictor + Joiner (3 networks) |
| Catalog models | Phi, Llama, Qwen (text LLMs) | No ASR models |

## Where Foundry Local Adds Value (Future)

For **downstream NLP** after speech transcription:

```
Mic → [NeMo Parakeet] → text → [Foundry Local: Phi-4] → structured commands
                                                        → entity extraction
                                                        → summarization
```

## Install Steps (When Ready)

```bash
# 1. trust-manager (cert-manager already installed)
sudo helm install trust-manager jetstack/trust-manager \
  --namespace cert-manager --version v0.7.1 \
  --set app.trust.namespace=cert-manager \
  --kubeconfig /etc/rancher/k3s/k3s.yaml

# 2. Foundry Local operator
sudo helm upgrade --install inference-operator \
  oci://mcr.microsoft.com/foundrylocalonazurelocal/helmcharts/helm/inference-operator \
  --version 0.0.1-prp.3 -n foundry-local-operator --create-namespace \
  --kubeconfig /etc/rancher/k3s/k3s.yaml

# 3. Deploy a text model (e.g., Phi-4)
cat <<EOF | sudo k3s kubectl apply -f -
apiVersion: foundrylocal.azure.com/v1
kind: ModelDeployment
metadata:
  name: phi-4-gpu
  namespace: foundry-local-operator
spec:
  displayName: "Phi-4 GPU"
  model:
    catalog:
      alias: Phi-4-cuda-gpu
  workloadType: generative
  compute: gpu
  replicas: 1
EOF
```

## Alternative: NVIDIA Triton

For true model-serving separation with NeMo ASR support, consider **NVIDIA Triton Inference Server** — it natively handles NeMo models, gRPC streaming, and the full ASR pipeline.
