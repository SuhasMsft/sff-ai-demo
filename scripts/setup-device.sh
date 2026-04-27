#!/bin/bash
# setup-device.sh — Reproducible device provisioning for SFF AI Demo
# Run this after a fresh Azure Linux 3.0 deployment or A/B swap
set -euo pipefail

echo "=== SFF AI Demo — Device Setup ==="

# 1. GPU device nodes (lost on every reboot — /dev is tmpfs)
echo "[1/7] Creating GPU device nodes + udev rule..."
sudo mknod -m 666 /dev/nvidia0 c 195 0 2>/dev/null || true
sudo mknod -m 666 /dev/nvidiactl c 195 255 2>/dev/null || true
# Install udev rule for persistence across reboots
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/99-nvidia-device-nodes.rules" ]; then
    sudo cp "$SCRIPT_DIR/99-nvidia-device-nodes.rules" /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    echo "udev rule installed for persistent GPU nodes"
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# 2. ALSA utils for audio diagnostics
echo "[2/7] Installing alsa-utils..."
sudo tdnf install -y alsa-utils 2>/dev/null || echo "alsa-utils already installed or unavailable"

# 3. Helm
echo "[3/7] Installing Helm..."
if ! command -v helm &>/dev/null; then
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
else
    echo "Helm already installed: $(helm version --short)"
fi

# 4. Disk-backed HuggingFace cache (NOT tmpfs!)
echo "[4/7] Creating disk-backed model cache..."
sudo mkdir -p /var/cache/hf-models
sudo chmod 777 /var/cache/hf-models

# 5. cert-manager (required for Foundry Local)
echo "[5/7] Installing cert-manager..."
if ! sudo k3s kubectl get namespace cert-manager &>/dev/null; then
    sudo helm repo add jetstack https://charts.jetstack.io --kubeconfig /etc/rancher/k3s/k3s.yaml
    sudo helm repo update --kubeconfig /etc/rancher/k3s/k3s.yaml
    sudo helm install cert-manager jetstack/cert-manager \
        --namespace cert-manager --create-namespace \
        --version v1.16.2 --set crds.enabled=true \
        --kubeconfig /etc/rancher/k3s/k3s.yaml
else
    echo "cert-manager namespace already exists"
fi

# 6. trust-manager (required for Foundry Local TLS)
echo "[6/7] Installing trust-manager..."
if ! sudo k3s kubectl get deployment trust-manager -n cert-manager &>/dev/null; then
    sudo helm install trust-manager jetstack/trust-manager \
        --namespace cert-manager --version v0.7.1 \
        --set app.trust.namespace=cert-manager \
        --kubeconfig /etc/rancher/k3s/k3s.yaml
else
    echo "trust-manager already installed"
fi

# 7. Clean up stale K3s pods (post-reboot artifacts)
echo "[7/7] Cleaning failed K3s pods..."
sudo k3s kubectl delete pods -A --field-selector=status.phase=Failed --grace-period=0 2>/dev/null || true

echo ""
echo "=== Setup complete ==="
echo "Next: cd /home/clouduser/demo/sff-ai-demo && docker compose up -d"
