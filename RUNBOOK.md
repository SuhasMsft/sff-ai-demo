# SFF AI Demo — Deployment Runbook

> **Audience**: Any engineer with SSH access to an Azure Local SFF device.
> **Repo**: https://github.com/SuhasMsft/sff-ai-demo
> **Time**: ~60-90 min (first deploy), ~10 min (redeploy)
> **All sections are idempotent** — safe to re-run if interrupted.

---

## Quick Start (experienced users)

> ⚠️ **Assumes GPU device nodes and nvidia-container-toolkit already configured.**
> If this is a fresh device, start at Section 0 instead.

```bash
# 1. SSH into device
az ssh arc --subscription <SUB_ID> --resource-group <RG> --name <ARC_MACHINE> --local-user clouduser --private-key-file ~/.ssh/<KEY>.pem -- -o StrictHostKeyChecking=no

# 2. Verify GPU + K3s (fail fast if not ready)
nvidia-smi >/dev/null 2>&1 && sudo k3s kubectl get runtimeclass nvidia >/dev/null 2>&1 && echo "GPU + K3s OK" || { echo "FAIL: Run Sections 1-2 first"; exit 1; }

# 3. Clone + setup
git clone https://github.com/SuhasMsft/sff-ai-demo.git && cd sff-ai-demo
chmod +x scripts/*.sh && sudo ./scripts/setup-device.sh

# 4. Build image + push to local registry
sudo docker run -d -p 5000:5000 --restart=always --name registry registry:2 2>/dev/null || true
sudo docker build -t mic-access:latest .
sudo docker tag mic-access:latest localhost:5000/mic-access:latest
sudo docker push localhost:5000/mic-access:latest

# 5. Prefetch models (inside container)
sudo docker run --rm -v /var/cache/hf-models:/hf-cache -e HF_HOME=/hf-cache --network host mic-access:latest python3 -c "
import nemo.collections.asr as nemo_asr; nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoTokenizer, AutoModelForCausalLM
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base'); AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('All models cached successfully')
"

# 6. Deploy to K3s
sudo k3s kubectl apply -f k8s/speech-recognizer.yaml && sudo k3s kubectl -n speech rollout status deployment/speech-recognizer --timeout=300s
```

---

## Section 0: Prerequisites Check

Run this preflight block — all checks must pass:

```bash
echo "=== SFF AI Demo Preflight ==="

# 1. Arc connectivity
echo -n "Arc agent: "; sudo azcmagent show 2>/dev/null | grep -E 'Agent Status' | awk -F: '{print $2}' || echo "NOT FOUND"

# 2. K3s cluster
echo -n "K3s node: "; sudo k3s kubectl get nodes --no-headers 2>/dev/null | awk '{print $1, $2}' || echo "NOT RUNNING"

# 3. GPU hardware
echo -n "NVIDIA GPU: "; lspci | grep -i nvidia | head -1 || echo "NOT FOUND"

# 4. GPU driver (if fails → Section 1)
echo -n "nvidia-smi: "; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "FAILED → run Section 1"

# 5. nvidia RuntimeClass (if fails → Section 2)
echo -n "RuntimeClass: "; sudo k3s kubectl get runtimeclass nvidia --no-headers 2>/dev/null | awk '{print $1}' || echo "MISSING → run Section 2"

# 6. Audio devices
echo -n "Audio: "; ls /dev/snd/pcmC*c 2>/dev/null | wc -l | xargs -I{} echo "{} capture devices" || echo "NONE"

# 7. Camera (optional — speech-only mode if absent)
echo -n "Camera: "; ls /dev/video* 2>/dev/null | wc -l | xargs -I{} echo "{} video devices" || echo "NONE (speech-only mode)"

# 8. Internet
echo -n "Internet: "; curl -so/dev/null -w'%{http_code}' https://huggingface.co 2>/dev/null || echo "OFFLINE"

# 9. Disk space
echo -n "Disk free: "; df -h / | awk 'NR==2{print $4, "available"}'

echo "=== Preflight Complete ==="
```

**Expected output** (all green):
```
Arc agent:  Connected
K3s node: 651441a2-... Ready
NVIDIA GPU: NVIDIA Corporation Device 28b0 (rev a1)
nvidia-smi: NVIDIA RTX 2000E Ada Generation, 580.105.08
RuntimeClass: nvidia
Audio: 1 capture devices
Camera: 2 video devices       (or "NONE (speech-only mode)" — OK)
Internet: 200
Disk free: 23G available
```

**If nvidia-smi says FAILED** → go to Section 1.
**If RuntimeClass says MISSING** → go to Section 2.
**If K3s not running** → `sudo systemctl restart k3s`
**If Arc disconnected** → `sudo azcmagent connect` or check network.

---

## Section 1: GPU Setup

> **Skip check**: `nvidia-smi >/dev/null 2>&1 && echo "Section 1 already done — skip to Section 2"`

### 1.1 Check kernel modules
```bash
lsmod | grep nvidia
```
**Expected**: `nvidia`, `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset` all listed.

If empty:
```bash
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

### 1.2 Create device nodes (if nvidia-smi fails)
```bash
# /dev is tmpfs — nodes vanish on reboot
sudo mknod -m 666 /dev/nvidia0 c 195 0 2>/dev/null
sudo mknod -m 666 /dev/nvidiactl c 195 255 2>/dev/null
nvidia-smi
```

### 1.3 Install udev rule for persistence
```bash
# From the repo:
sudo cp scripts/99-nvidia-device-nodes.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
echo "Udev rule installed — GPU nodes will persist across reboots"
```

### 1.4 Verify
```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# Expected: NVIDIA RTX 2000E Ada Generation, 580.105.08, 16380 MiB
```

---

## Section 2: NVIDIA Container Toolkit + K3s GPU

> **Skip check**: `sudo k3s kubectl get runtimeclass nvidia >/dev/null 2>&1 && echo "Section 2 already done — skip to Section 3"`

### 2.1 Install nvidia-container-toolkit

```bash
# Prerequisites
sudo tdnf install -y ca-certificates curl
sudo update-ca-trust

# Check if already installed
nvidia-ctk --version 2>/dev/null && echo "Already installed — skip to 2.2"

# Add NVIDIA's official RPM repo (works with tdnf)
sudo curl -fsSL \
    https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    -o /etc/yum.repos.d/nvidia-container-toolkit.repo

sudo tdnf clean all && sudo tdnf makecache
sudo tdnf install -y nvidia-container-toolkit

# Verify
nvidia-ctk --version
/usr/bin/nvidia-container-runtime --version
```

**If tdnf install fails** (package not found — fallback to direct RPMs):
```bash
mkdir -p ~/nvidia-toolkit-rpms && cd ~/nvidia-toolkit-rpms
ARCH=$(uname -m)
for pkg in libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit-base nvidia-container-toolkit; do
    curl -fLO "https://nvidia.github.io/libnvidia-container/stable/rpm/${ARCH}/${pkg}-1.17.4-1.${ARCH}.rpm"
done
sudo tdnf localinstall -y ./*.rpm
nvidia-ctk --version
```

### 2.2 Configure for K3s containerd

> ⚠️ **CRITICAL K3s RULES**:
> - Do NOT run `nvidia-ctk runtime configure --runtime=containerd` (targets wrong config path)
> - Do NOT create `config.toml.tmpl` (replaces K3s's entire generated config, breaks CNI)
> - K3s v1.34+ uses containerd config **version 3**

```bash
# Restart K3s — it auto-detects nvidia-container-runtime if in PATH
# ⚠️ This briefly disrupts all pods on this node
sudo systemctl restart k3s
sleep 10

# Verify K3s found the nvidia runtime
sudo grep -n nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

**If grep shows nothing** (K3s didn't auto-detect) — add a v3 drop-in:
```bash
sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.d

sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.d/99-nvidia.toml >/dev/null <<'EOF'
[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']
  runtime_type = "io.containerd.runc.v2"

[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia'.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = true
EOF

sudo systemctl restart k3s
sudo grep nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml
```

### 2.3 Create NVIDIA RuntimeClass + Device Plugin
```bash
# RuntimeClass (check first)
sudo k3s kubectl get runtimeclass nvidia 2>/dev/null || \
cat <<EOF | sudo k3s kubectl apply -f -
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: nvidia
handler: nvidia
EOF

# Device Plugin (exposes nvidia.com/gpu resource to K3s scheduler)
sudo k3s kubectl apply -f \
    https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml

# Wait for device plugin to be ready
sudo k3s kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s
```

### 2.4 Verify GPU visible to K3s
```bash
sudo k3s kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\\.com/gpu
# Expected: NAME=651441a2-...  GPUS=1
```

### 2.5 Smoke test — GPU in a K3s pod
```bash
cat <<EOF | sudo k3s kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  restartPolicy: Never
  runtimeClassName: nvidia
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

# Wait for completion (non-interactive, Arc SSH safe)
sleep 30
sudo k3s kubectl logs gpu-test 2>/dev/null || echo "Pod not ready yet — wait and retry"
# Expected: Shows your GPU name + driver version
sudo k3s kubectl delete pod gpu-test --ignore-not-found
```

---

## Section 3: Clone Repo + Setup

> **Skip check**: `ls ~/sff-ai-demo/access_mic.py >/dev/null 2>&1 && echo "Section 3 already done — skip to Section 4"`

### 3.1 Clone
```bash
cd /home/clouduser
git clone https://github.com/SuhasMsft/sff-ai-demo.git
cd sff-ai-demo
```

### 3.2 Run device setup
```bash
chmod +x scripts/*.sh
sudo ./scripts/setup-device.sh
```

This installs: GPU device nodes + udev rule, alsa-utils, Helm, cert-manager, disk-backed HF cache directory.

### 3.3 Pre-download models (inside container — avoids host Python deps)
```bash
# First build the image (needed for prefetch)
# NOTE: First build takes 20-45 minutes. This is normal.
# For Arc SSH: use nohup to prevent timeout on long builds
nohup sudo docker build -t mic-access:latest . > /tmp/docker-build.log 2>&1 &
echo "Building... tail -f /tmp/docker-build.log to watch progress"
wait  # Wait for build to finish
tail -5 /tmp/docker-build.log  # Check result

# Start local registry if not running
sudo docker run -d -p 5000:5000 --restart=always --name registry registry:2 2>/dev/null || true

# Then prefetch all 3 models using the container's Python environment
sudo mkdir -p /var/cache/hf-models && sudo chmod 777 /var/cache/hf-models

sudo docker run --rm \
    -v /var/cache/hf-models:/hf-cache \
    -e HF_HOME=/hf-cache \
    --network host \
    mic-access:latest \
    python3 -c "
import nemo.collections.asr as nemo_asr
print('Downloading Parakeet...')
nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoTokenizer, AutoModelForCausalLM
print('Downloading Grounding-DINO...')
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Downloading Qwen...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('All 3 models cached successfully')
"
```

### 3.4 Verify cache
```bash
du -sh /var/cache/hf-models/
# Expected: ~5-6 GB
```

---

## Section 4: Build + Load Container Image

### Option A: Docker → local registry → K3s pulls
```bash
# Tag and push
sudo docker tag mic-access:latest localhost:5000/mic-access:latest
sudo docker push localhost:5000/mic-access:latest

# Verify
curl -s http://localhost:5000/v2/_catalog
# Expected: {"repositories":["mic-access"]}
```

**If localhost:5000 refuses connection** — start the registry:
```bash
sudo docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

### Option B: Direct K3s import (no registry needed)
```bash
# Export image
sudo docker save mic-access:latest -o /tmp/mic-access.tar

# Import into K3s containerd
sudo k3s ctr images import /tmp/mic-access.tar

# Verify
sudo k3s ctr images ls | grep mic-access

# Update manifest to use local image (not registry)
# In k8s/speech-recognizer.yaml, change:
#   image: localhost:5000/mic-access:latest
# To:
#   image: docker.io/library/mic-access:latest
#   imagePullPolicy: Never
```

---

## Section 5: Deploy to K3s

```bash
# Apply manifest
sudo k3s kubectl apply -f k8s/speech-recognizer.yaml

# Watch pod come up
sudo k3s kubectl -n speech get pods -w
# Wait until STATUS = Running (may take 30-60s for model loading)

# Check logs
sudo k3s kubectl -n speech logs -f deployment/speech-recognizer
```

**Expected log output:**
```
=== Startup preflight ===
ALSA path found: /dev/snd (12 nodes)
PortAudio inputs detected: 1
  Input 1: USB PnP Audio Device: Audio (hw:1,0) (channels=1, sample_rate=48000.0)
V4L2 video devices found: 2 (/dev/video0, /dev/video1)
DNS check: huggingface.co resolved
HTTPS check: huggingface.co reachable (status=200)
=== Preflight complete ===
✔ Check CUDA availability succeeded in 0.01s
Using device: cuda:0
✔ Prepare camera succeeded in 0.50s
✔ Prepare microphone succeeded in 0.01s
✔ Load vision model: grounding-dino-base succeeded in 5.00s
✔ Load language model: qwen2.5-0.5b-instruct succeeded in 3.00s
✔ Load speech model: parakeet-tdt-0.6b-v2 succeeded in 8.00s
Listening...
Listening... Press Ctrl+C to stop.
```

### Verify GPU usage
```bash
nvidia-smi
# Expected: 2 processes using ~3-5 GB VRAM (parakeet + grounding-dino)
```

---

## Section 6: Verification

### 6.1 Preflight passed?
```bash
sudo k3s kubectl -n speech logs deployment/speech-recognizer | grep "Preflight complete"
# Expected: "=== Preflight complete ==="
```

### 6.2 All models loaded?
```bash
sudo k3s kubectl -n speech logs deployment/speech-recognizer | grep "succeeded"
# Expected: 5 lines (CUDA, camera, mic, DINO, Qwen, Parakeet)
```

### 6.3 Speech recognition test
Speak "hello world" into the USB mic. Check logs:
```bash
sudo k3s kubectl -n speech logs deployment/speech-recognizer --tail=5
# Expected: Heard: 'hello world'
#           Interpreted: 'hello world'
```

### 6.4 Vision test (if camera present)
Speak "show me the red ball" with a red ball in camera view:
```bash
sudo k3s kubectl -n speech logs deployment/speech-recognizer --tail=10
# Expected: Heard: 'show me the red ball'
#           Interpreted: 'red ball'
#           Looking for: 'red ball'...
#           Detected 'ball' at (320, 240) confidence=0.85
```

### 6.5 Resource check
```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
# Expected: ~3000-5000 MiB, 0-30%

sudo k3s kubectl -n speech top pod
# Expected: CPU ~100-500m idle, Memory ~4-6Gi
```

---

## Section 7: Troubleshooting

### Pod CrashLoopBackOff
```bash
sudo k3s kubectl -n speech describe pod -l app=speech-recognizer | tail -20
sudo k3s kubectl -n speech logs deployment/speech-recognizer --previous
```
| Error in logs | Cause | Fix |
|---------------|-------|-----|
| `FATAL: GPU required but not found` | /dev/nvidia0 missing | Section 1 (GPU setup) |
| `RuntimeError: No microphone found` | /dev/snd not mounted | Check manifest hostPath |
| `OOMKilled` | Container exceeded 12Gi | Increase memory limit or use FP16 |
| `nvidia RuntimeClass not found` | Section 2 not done | Create RuntimeClass |

### No audio detected
```bash
# Check from inside pod
sudo k3s kubectl -n speech exec deployment/speech-recognizer -- ls -la /dev/snd/
sudo k3s kubectl -n speech exec deployment/speech-recognizer -- python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

### No camera (speech-only mode)
This is **expected and handled** — the app runs in speech-only mode. Logs will show:
```
WARNING: No camera found. Vision disabled — speech-only mode.
```

### Arc SSH timeout
```bash
# Device is under heavy load. Wait 5 min or:
az connectedmachine show --subscription <SUB> --resource-group <RG> --name <MACHINE> --query status -o tsv
# If "Disconnected" for >15 min, device may need physical reboot
```

---

## Section 8: Teardown

```bash
# Remove deployment
sudo k3s kubectl delete -f k8s/speech-recognizer.yaml

# Optional: remove images
sudo k3s ctr images rm docker.io/library/mic-access:latest
sudo docker rmi mic-access:latest localhost:5000/mic-access:latest 2>/dev/null

# Optional: clean model cache (saves ~5GB disk)
sudo rm -rf /var/cache/hf-models/*
```

---

## Development Cycle (code changes)

### Docker path
```bash
cd /home/clouduser/sff-ai-demo
# Edit code...
sudo docker build -t mic-access:latest .
sudo docker tag mic-access:latest localhost:5000/mic-access:latest
sudo docker push localhost:5000/mic-access:latest
sudo k3s kubectl -n speech rollout restart deployment/speech-recognizer
sudo k3s kubectl -n speech logs -f deployment/speech-recognizer
```

### Rollback
```bash
sudo k3s kubectl -n speech rollout undo deployment/speech-recognizer
```

---

## Appendix: SSH Access

### Prerequisites (on your laptop)
```bash
# 1. Install Azure CLI SSH extension
az extension add --name ssh

# 2. Login
az login

# 3. Fix PEM key permissions (Windows only)
icacls "C:\path\to\key.pem" /inheritance:r /grant:r "%USERNAME%:(R)"
```

### Connect
```bash
az ssh arc \
    --subscription <SUBSCRIPTION_ID> \
    --resource-group <RESOURCE_GROUP> \
    --name <ARC_MACHINE_NAME> \
    --local-user clouduser \
    --private-key-file ~/.ssh/<KEY>.pem \
    -- -o StrictHostKeyChecking=no
```

To find your device's identifiers:
```bash
az connectedmachine list --query "[?contains(name,'sff')].{name:name, rg:resourceGroup, sub:subscriptionId}" -o table
```
