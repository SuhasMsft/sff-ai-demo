#!/bin/bash
# run-runbook.sh — Automated SFF AI Demo deployment with progress logging
# Usage: sudo ./scripts/run-runbook.sh [--skip-build] [--skip-prefetch]
set -euo pipefail

LOG="/var/log/sff-ai-demo-deploy.log"
STEP=0
TOTAL=12
START_TIME=$(date +%s)

log() {
    local msg="[$(date '+%H:%M:%S')] [Step $STEP/$TOTAL] $1"
    echo "$msg" | tee -a "$LOG"
}

pass() { echo "  ✅ $1" | tee -a "$LOG"; }
fail() { echo "  ❌ $1" | tee -a "$LOG"; }
warn() { echo "  ⚠️  $1" | tee -a "$LOG"; }
elapsed() { echo "  ⏱  $(( $(date +%s) - START_TIME ))s elapsed" | tee -a "$LOG"; }

SKIP_BUILD=false
SKIP_PREFETCH=false
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --skip-prefetch) SKIP_PREFETCH=true ;;
    esac
done

echo "" | tee "$LOG"
echo "╔══════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║   SFF AI Demo — Automated Deployment         ║" | tee -a "$LOG"
echo "║   $(date '+%Y-%m-%d %H:%M:%S')                        ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# ─── STEP 1: PREFLIGHT ─────────────────────────────────────
STEP=1; log "PREFLIGHT CHECKS"
PREFLIGHT_PASS=true

echo -n "  Arc agent: " | tee -a "$LOG"
ARC=$(sudo azcmagent show 2>/dev/null | grep 'Agent Status' | awk -F: '{print $2}' | xargs)
if [ "$ARC" = "Connected" ]; then pass "Connected"; else fail "$ARC"; PREFLIGHT_PASS=false; fi

echo -n "  K3s: " | tee -a "$LOG"
K3S=$(sudo k3s kubectl get nodes --no-headers 2>/dev/null | awk '{print $2}')
if [ "$K3S" = "Ready" ]; then pass "Ready"; else warn "Not Ready ($K3S) — will attempt restart"; fi

echo -n "  GPU hardware: " | tee -a "$LOG"
GPU_HW=$(lspci | grep -i nvidia | head -1)
if [ -n "$GPU_HW" ]; then pass "$GPU_HW"; else fail "No NVIDIA GPU found"; PREFLIGHT_PASS=false; fi

echo -n "  nvidia-smi: " | tee -a "$LOG"
if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    pass "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
    warn "FAILED — will fix in Step 2"
fi

echo -n "  Audio: " | tee -a "$LOG"
AUDIO_COUNT=$(ls /dev/snd/pcmC*c 2>/dev/null | wc -l)
if [ "$AUDIO_COUNT" -gt 0 ]; then pass "$AUDIO_COUNT capture device(s)"; else fail "No audio capture devices"; fi

echo -n "  Camera: " | tee -a "$LOG"
CAM_COUNT=$(ls /dev/video* 2>/dev/null | wc -l || echo 0)
if [ "$CAM_COUNT" -gt 0 ]; then pass "$CAM_COUNT video device(s)"; else warn "None — speech-only mode"; fi

echo -n "  Internet: " | tee -a "$LOG"
HTTP=$(curl -so/dev/null -w'%{http_code}' https://huggingface.co 2>/dev/null || echo "000")
if [ "$HTTP" = "200" ]; then pass "HuggingFace reachable"; else fail "HTTP $HTTP"; fi

echo -n "  Disk: " | tee -a "$LOG"
DISK_FREE=$(df -BG / | awk 'NR==2{print $4}')
pass "$DISK_FREE free"

elapsed

# ─── STEP 2: GPU DEVICE NODES ──────────────────────────────
STEP=2; log "GPU DEVICE NODES"
if nvidia-smi >/dev/null 2>&1; then
    pass "nvidia-smi already working — skipping"
else
    log "Loading nvidia modules..."
    sudo modprobe nvidia 2>/dev/null || true
    sudo modprobe nvidia_uvm 2>/dev/null || true
    log "Creating device nodes..."
    sudo mknod -m 666 /dev/nvidia0 c 195 0 2>/dev/null || true
    sudo mknod -m 666 /dev/nvidiactl c 195 255 2>/dev/null || true
    if nvidia-smi >/dev/null 2>&1; then
        pass "nvidia-smi now working"
    else
        fail "nvidia-smi still fails — check driver"
        exit 1
    fi
fi

# Install udev rule for persistence
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/99-nvidia-device-nodes.rules" ]; then
    sudo cp "$SCRIPT_DIR/99-nvidia-device-nodes.rules" /etc/udev/rules.d/ 2>/dev/null || true
    sudo udevadm control --reload-rules 2>/dev/null || true
    pass "udev rule installed"
fi
elapsed

# ─── STEP 3: NVIDIA CONTAINER TOOLKIT ──────────────────────
STEP=3; log "NVIDIA CONTAINER TOOLKIT"
if nvidia-ctk --version >/dev/null 2>&1; then
    pass "Already installed: $(nvidia-ctk --version 2>&1 | head -1)"
else
    log "Installing from NVIDIA RPM repo..."
    sudo tdnf install -y ca-certificates curl 2>/dev/null || true
    sudo curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
        -o /etc/yum.repos.d/nvidia-container-toolkit.repo 2>/dev/null
    sudo tdnf clean all >/dev/null 2>&1
    sudo tdnf makecache >/dev/null 2>&1
    if sudo tdnf install -y nvidia-container-toolkit 2>/dev/null; then
        pass "Installed: $(nvidia-ctk --version 2>&1 | head -1)"
    else
        warn "tdnf install failed — attempting direct RPM download..."
        ARCH=$(uname -m)
        mkdir -p /tmp/nvidia-toolkit-rpms && cd /tmp/nvidia-toolkit-rpms
        for pkg in libnvidia-container1 libnvidia-container-tools nvidia-container-toolkit-base nvidia-container-toolkit; do
            curl -fLO "https://nvidia.github.io/libnvidia-container/stable/rpm/${ARCH}/${pkg}-1.17.4-1.${ARCH}.rpm" 2>/dev/null
        done
        sudo tdnf localinstall -y ./*.rpm 2>/dev/null
        cd -
        if nvidia-ctk --version >/dev/null 2>&1; then
            pass "Installed via RPM: $(nvidia-ctk --version 2>&1 | head -1)"
        else
            fail "Could not install nvidia-container-toolkit"
            exit 1
        fi
    fi
fi
elapsed

# ─── STEP 4: K3S GPU RUNTIME ───────────────────────────────
STEP=4; log "K3S GPU RUNTIME CONFIGURATION"
if sudo k3s kubectl get runtimeclass nvidia >/dev/null 2>&1; then
    pass "RuntimeClass 'nvidia' already exists — skipping"
else
    log "Restarting K3s to detect nvidia runtime..."
    sudo systemctl restart k3s
    sleep 15
    log "Checking if K3s auto-detected nvidia..."
    if sudo grep -q nvidia /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null; then
        pass "K3s auto-detected nvidia runtime"
    else
        log "Adding containerd v3 drop-in..."
        sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.d
        sudo tee /var/lib/rancher/k3s/agent/etc/containerd/config-v3.toml.d/99-nvidia.toml >/dev/null <<'DROPEOF'
[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia']
  runtime_type = "io.containerd.runc.v2"
[plugins.'io.containerd.cri.v1.runtime'.containerd.runtimes.'nvidia'.options]
  BinaryName = "/usr/bin/nvidia-container-runtime"
  SystemdCgroup = true
DROPEOF
        sudo systemctl restart k3s
        sleep 15
        pass "Drop-in installed"
    fi

    log "Creating RuntimeClass..."
    printf '%s\n' 'apiVersion: node.k8s.io/v1' 'kind: RuntimeClass' 'metadata:' '  name: nvidia' 'handler: nvidia' | sudo k3s kubectl apply -f -

    log "Installing NVIDIA device plugin..."
    sudo k3s kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml 2>/dev/null
    sudo k3s kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset --timeout=180s 2>/dev/null || warn "Device plugin not ready yet"

    log "Verifying GPU visible to K3s..."
    GPUS=$(sudo k3s kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
    if [ "$GPUS" = "1" ]; then pass "GPU count: $GPUS"; else warn "GPU count: $GPUS (expected 1)"; fi
fi
elapsed

# ─── STEP 5: SETUP + CACHE ─────────────────────────────────
STEP=5; log "DEVICE SETUP"
sudo mkdir -p /var/cache/hf-models && sudo chmod 777 /var/cache/hf-models
sudo tdnf install -y alsa-utils 2>/dev/null || true
pass "Cache dir + alsa-utils ready"
elapsed

# ─── STEP 6: CLONE REPO ────────────────────────────────────
STEP=6; log "CLONE REPO"
if [ -f /home/clouduser/sff-ai-demo/access_mic.py ]; then
    log "Repo already exists — pulling latest..."
    cd /home/clouduser/sff-ai-demo && git pull 2>/dev/null || true
    pass "Updated"
else
    cd /home/clouduser
    git clone https://github.com/SuhasMsft/sff-ai-demo.git 2>/dev/null
    cd sff-ai-demo
    pass "Cloned"
fi
elapsed

# ─── STEP 7: BUILD IMAGE ───────────────────────────────────
STEP=7; log "BUILD CONTAINER IMAGE"
cd /home/clouduser/sff-ai-demo
if [ "$SKIP_BUILD" = true ]; then
    warn "Skipped (--skip-build flag)"
else
    log "Building image (this takes 20-45 min on first run)..."
    sudo docker build -t mic-access:latest . 2>&1 | tail -5 | tee -a "$LOG"
    pass "Image built"

    log "Pushing to local registry..."
    sudo docker run -d -p 5000:5000 --restart=always --name registry registry:2 2>/dev/null || true
    sudo docker tag mic-access:latest localhost:5000/mic-access:latest
    sudo docker push localhost:5000/mic-access:latest 2>&1 | tail -3 | tee -a "$LOG"
    pass "Pushed to localhost:5000"
fi
elapsed

# ─── STEP 8: PREFETCH MODELS ───────────────────────────────
STEP=8; log "PREFETCH AI MODELS"
CACHE_SIZE=$(du -sm /var/cache/hf-models/ 2>/dev/null | awk '{print $1}')
if [ "${CACHE_SIZE:-0}" -gt 3000 ] && [ "$SKIP_PREFETCH" = false ]; then
    pass "Models already cached (${CACHE_SIZE}MB) — skipping"
elif [ "$SKIP_PREFETCH" = true ]; then
    warn "Skipped (--skip-prefetch flag)"
else
    log "Downloading 3 models (~5GB total)..."
    sudo docker run --rm \
        -v /var/cache/hf-models:/hf-cache \
        -e HF_HOME=/hf-cache \
        --network host \
        mic-access:latest \
        python3 -c "
import nemo.collections.asr as nemo_asr
print('Downloading Parakeet TDT 0.6B...')
nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoTokenizer, AutoModelForCausalLM
print('Downloading Grounding-DINO...')
AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Downloading Qwen 2.5...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('All 3 models cached')
" 2>&1 | grep -E 'Downloading|cached|Error' | tee -a "$LOG"
    CACHE_SIZE=$(du -sm /var/cache/hf-models/ 2>/dev/null | awk '{print $1}')
    pass "Models cached (${CACHE_SIZE}MB)"
fi
elapsed

# ─── STEP 9: DEPLOY TO K3S ─────────────────────────────────
STEP=9; log "DEPLOY TO K3S"
cd /home/clouduser/sff-ai-demo
sudo k3s kubectl apply -f k8s/speech-recognizer.yaml 2>&1 | tee -a "$LOG"
log "Waiting for pod to be Running..."
sudo k3s kubectl -n speech rollout status deployment/speech-recognizer --timeout=300s 2>&1 | tee -a "$LOG"
pass "Deployed"
elapsed

# ─── STEP 10: VERIFY ───────────────────────────────────────
STEP=10; log "VERIFICATION"
sleep 10

echo -n "  Pod status: " | tee -a "$LOG"
POD_STATUS=$(sudo k3s kubectl -n speech get pods --no-headers 2>/dev/null | awk '{print $3}')
if [ "$POD_STATUS" = "Running" ]; then pass "Running"; else fail "$POD_STATUS"; fi

echo -n "  GPU VRAM: " | tee -a "$LOG"
VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
pass "$VRAM"

echo -n "  App status: " | tee -a "$LOG"
LISTENING=$(sudo k3s kubectl -n speech logs deployment/speech-recognizer --tail=5 2>/dev/null | grep -c "Listening" || echo 0)
if [ "$LISTENING" -gt 0 ]; then pass "Listening for speech"; else warn "Not yet listening — check logs"; fi

echo "" | tee -a "$LOG"
log "LAST 10 LOG LINES:"
sudo k3s kubectl -n speech logs deployment/speech-recognizer --tail=10 2>/dev/null | tee -a "$LOG"
elapsed

# ─── STEP 11: RESOURCE CHECK ───────────────────────────────
STEP=11; log "RESOURCE CHECK"
echo -n "  RAM: " | tee -a "$LOG"; free -h | awk 'NR==2{print $3, "used /", $2, "total"}' | tee -a "$LOG"
echo -n "  GPU: " | tee -a "$LOG"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | tee -a "$LOG"
echo -n "  Disk: " | tee -a "$LOG"; df -h / | awk 'NR==2{print $3, "used /", $2, "total"}' | tee -a "$LOG"
echo -n "  K3s pods: " | tee -a "$LOG"; sudo k3s kubectl get pods -A --no-headers 2>/dev/null | awk '{print $4}' | sort | uniq -c | sort -rn | tr '\n' ' ' | tee -a "$LOG"; echo "" | tee -a "$LOG"
elapsed

# ─── STEP 12: DONE ─────────────────────────────────────────
STEP=12; log "DEPLOYMENT COMPLETE"
TOTAL_TIME=$(( $(date +%s) - START_TIME ))
echo "" | tee -a "$LOG"
echo "╔══════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║   ✅ SFF AI Demo deployed successfully        ║" | tee -a "$LOG"
echo "║   Total time: ${TOTAL_TIME}s                           ║" | tee -a "$LOG"
echo "║   Log: $LOG              ║" | tee -a "$LOG"
echo "║                                              ║" | tee -a "$LOG"
echo "║   View logs:                                 ║" | tee -a "$LOG"
echo "║   sudo k3s kubectl -n speech logs -f \\       ║" | tee -a "$LOG"
echo "║     deployment/speech-recognizer             ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════╝" | tee -a "$LOG"
