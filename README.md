# SFF AI Demo — Collaborative Robot on Azure Local

Voice-controlled object detection demo running 3 AI models on an Azure Local SFF edge device. Say "show me the blue cube" → the system transcribes speech, extracts the object, and detects it via camera.

> **Upstream**: Containerized + hardened version of [cosmosdarwin/cobotpoc](https://github.com/cosmosdarwin/cobotpoc)

## Demo Flow

```
        "show me the blue cube"
              │
    ┌─────────▼──────────┐
    │  USB Microphone     │  48kHz mono, continuous listening
    │  ↓ energy threshold │  segments speech from silence
    │  ↓ downsample 3x    │
    └─────────┬──────────┘
              │ audio buffer
    ┌─────────▼──────────┐
    │  Parakeet TDT 0.6B │  NVIDIA NeMo ASR (GPU)
    │  → "show me the    │  ~1-3 sec per utterance on GPU
    │     blue cube"     │
    └─────────┬──────────┘
              │ raw text
    ┌─────────▼──────────┐
    │  Qwen 2.5 0.5B     │  HuggingFace Transformers (CPU)
    │  → "blue cube"     │  extracts object phrase from speech
    └─────────┬──────────┘
              │ search query
    ┌─────────▼──────────┐
    │  Grounding-DINO     │  Zero-shot object detection (GPU)
    │  + USB Camera       │  finds "blue cube" in camera frame
    │  + Color matching   │  HSL hue comparison for color
    │  → (x=320, y=240)  │  returns center coordinates
    └────────────────────┘
```

## AI Models

| Model | Task | Size | Device | VRAM |
|-------|------|------|--------|------|
| [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Speech → Text | 600M params | GPU | ~1.5 GB |
| [Grounding-DINO-Base](https://huggingface.co/IDEA-Research/grounding-dino-base) | Image + Text → Bounding Box | 172M params | GPU | ~2 GB |
| [Qwen 2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Text → Object Extraction | 500M params | CPU | 0 |
| **Total** | | | | **~3.5 GB / 16 GB** |

## Hardware

| Component | Spec |
|-----------|------|
| Device | Lenovo SE100 (Azure Local SFF) |
| CPU | Intel Core Ultra 5 225H (14 cores) |
| RAM | 29 GB (no swap) |
| GPU | NVIDIA RTX 2000E Ada Generation (16 GB VRAM) |
| OS | Microsoft Azure Linux 3.0 |
| Audio | USB PnP microphone (48 kHz mono) |
| Camera | UVC-compliant USB camera |
| Robot | MyCobot 280 M5 (planned) |

## Quick Start

```bash
# 1. Run device setup (GPU nodes, Helm, cert-manager, ALSA tools)
chmod +x scripts/setup-device.sh && ./scripts/setup-device.sh

# 2. Pre-download models (avoids long cold start)
chmod +x scripts/prefetch-models.sh && ./scripts/prefetch-models.sh

# 3. Build and run
docker compose up -d

# 4. Check logs — should show "Listening..."
docker logs -f mic-access
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│            Ubuntu 24.04 Container                 │
│  (on Azure Linux 3.0 host via Docker/K3s)        │
│                                                   │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────┐  │
│  │ Parakeet    │ │ Grounding-   │ │ Qwen 2.5  │  │
│  │ TDT 0.6B   │ │ DINO-Base    │ │ 0.5B      │  │
│  │ (ASR, GPU)  │ │ (Vision,GPU) │ │ (LLM,CPU) │  │
│  └──────┬──────┘ └──────┬───────┘ └─────┬─────┘  │
│         │               │               │         │
│  ┌──────┴───────────────┴───────────────┴──────┐  │
│  │         Orchestrator (state machine)         │  │
│  │  LISTENING → transcribe → extract → LOOKING  │  │
│  │  LOOKING → detect → found (x,y) → LISTENING  │  │
│  └─────────────────────────────────────────────┘  │
│         │                    │                     │
│    /dev/snd             /dev/video*                │
│    (USB mic)            (USB camera)               │
└──────────────────────────────────────────────────┘
```

## Resource Limits

| Resource | Limit | Why |
|----------|-------|-----|
| Memory | 12 GB (8 GB reserved) | 3 models + PyTorch + inference headroom |
| CPU | 4 cores | Leaves 10 cores for K3s, Arc, system |
| GPU | 1 × NVIDIA | All vision + speech models share VRAM |
| Restart | unless-stopped | Auto-recover from OOM |
| HF cache | Disk-backed volume | Prevents tmpfs double-storage (2.5GB RAM waste) |

## Improvements Over Upstream (cobotpoc)

| Feature | cobotpoc | sff-ai-demo |
|---------|----------|-------------|
| GPU auto-detect | ❌ Hardcodes CPU | ✅ `torch.cuda.is_available()` with fallback |
| Audio buffer | ❌ Unbounded list (memory leak) | ✅ `deque(maxlen=300)` (~30s cap) |
| Startup preflight | ❌ None | ✅ ALSA, PortAudio, camera, HuggingFace diagnostics |
| Virtual device filter | ❌ Picks PulseAudio monitors | ✅ Skips >8-channel virtual devices |
| Containerized | ❌ Bare-metal Ubuntu only | ✅ Docker + K3s + resource limits |
| Resource limits | ❌ None | ✅ Memory, CPU, restart policy |
| Camera degradation | ❌ Crashes if no camera | ✅ Falls back to speech-only mode |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ubuntu 24.04 container** (not Azure Linux) | `portaudio19-dev` and `ffmpeg` absent from Azure Linux repos |
| **GPU inference** (not CPU) | CPU pins all 14 cores for 30-60s; GPU does 1-3s |
| **Disk-backed HF cache** | `/tmp` is tmpfs on Azure Linux — model cached in RAM twice |
| **Single container** (not Foundry Local) | Qwen 0.5B too small for Foundry overhead; co-locate all 3 models |
| **Foundry Local deferred** | Only supports ONNX; Parakeet TDT decoder can't be exported |

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- NVIDIA GPU driver loaded (`nvidia-smi` working)
- `/dev/snd` audio devices + `/dev/video*` camera present

## Files

| File | Purpose |
|------|---------|
| `access_mic.py` | Main app — speech + vision + LLM state machine |
| `Dockerfile` | Ubuntu 24.04 + PortAudio + OpenCV + NeMo + Transformers |
| `docker-compose.yml` | GPU runtime, resource limits, device passthrough |
| `requirements.txt` | Python dependencies (11 packages) |
| `Dockerfile.old` | Previous Azure Linux attempt (archived) |

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/DEVICE-SETUP.md](docs/DEVICE-SETUP.md) | Device provisioning, SSH access, known issues |
| [docs/K3S-DEPLOYMENT.md](docs/K3S-DEPLOYMENT.md) | Running on K3s instead of Docker |
| [docs/FOUNDRY-LOCAL.md](docs/FOUNDRY-LOCAL.md) | Foundry Local analysis — ONNX constraint, future use |
| [k8s/speech-recognizer.yaml](k8s/speech-recognizer.yaml) | K3s deployment manifest (GPU + PVC) |
| [scripts/setup-device.sh](scripts/setup-device.sh) | Automated device setup (idempotent) |
| [scripts/prefetch-models.sh](scripts/prefetch-models.sh) | Pre-download all 3 model weights |
| [scripts/99-nvidia-device-nodes.rules](scripts/99-nvidia-device-nodes.rules) | udev rule for persistent GPU nodes |

## Remaining Work

| # | Item | Status | Needs Device? |
|---|------|:------:|:-------------:|
| 1 | `nvidia-container-toolkit` install | ⏳ Pending | Yes |
| 2 | `trust-manager` install (for future Foundry) | ⏳ Pending | Yes |
| 3 | Reboot test (GPU + K3s + Arc survive) | ⏳ Pending | Yes |
| 4 | K3s ResourceQuota + PriorityClass tuning | ⏳ Pending | Yes |
| 5 | End-to-end soak test (30 min) | ⏳ Pending | Yes |
| 6 | MyCobot 280 robot arm integration | 🔮 Future | Yes |
| 7 | VS Code Remote SSH config for dev access | ⏳ Pending | Client-side |

## Credits

- Speech model: [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- Vision model: [Grounding-DINO-Base](https://huggingface.co/IDEA-Research/grounding-dino-base)
- Language model: [Qwen 2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- Cobot POC: [Cosmos Darwin](https://github.com/cosmosdarwin/cobotpoc)
- Containerization: Roycey Cheeran
- GPU/resource optimization + hardening: Suhas Prakash
