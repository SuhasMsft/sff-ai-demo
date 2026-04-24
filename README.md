# SFF Speech Container — Azure Local Robotic Arms Demo

Speech-to-text container for the Azure Local SFF robotic arms demo. Runs [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) on an edge device with GPU acceleration.

> **Upstream**: This is the containerized version of the speech component from [cosmosdarwin/cobotpoc](https://github.com/cosmosdarwin/cobotpoc).

## Hardware

| Component | Spec |
|-----------|------|
| Device | Lenovo SE100 (Azure Local SFF) |
| CPU | Intel Core Ultra 5 225H (14 cores) |
| RAM | 29 GB |
| GPU | NVIDIA RTX 2000E Ada Generation (16 GB VRAM) |
| OS | Microsoft Azure Linux 3.0 |
| Audio | USB PnP microphone (48 kHz mono) |

## Quick Start

```bash
# 1. Create disk-backed model cache (NOT tmpfs!)
sudo mkdir -p /var/cache/hf-models && sudo chmod 777 /var/cache/hf-models

# 2. Create GPU device nodes (if missing after reboot)
sudo mknod -m 666 /dev/nvidia0 c 195 0 2>/dev/null
sudo mknod -m 666 /dev/nvidiactl c 195 255 2>/dev/null

# 3. Build and run
docker compose up -d

# 4. Check logs
docker logs -f mic-access
```

## Architecture

```
USB Mic (48kHz) → Container (Ubuntu 24.04)
                    ├── PortAudio → sounddevice
                    ├── NeMo Parakeet TDT 0.6B → GPU inference
                    └── Speech text output
```

The container:
- Captures audio via `/dev/snd` passthrough
- Auto-selects the best USB/mono microphone
- Downloads and caches the NeMo model on first run
- Runs inference on GPU (CUDA) with CPU fallback
- Segments speech by energy threshold, transcribes utterances

## Resource Limits

The `docker-compose.yml` enforces safe resource limits to prevent the container from starving system services (K3s, Arc agent):

| Resource | Limit | Why |
|----------|-------|-----|
| Memory | 8 GB (6 GB reserved) | Model ~3 GB + PyTorch + inference headroom |
| CPU | 4 cores | Leaves 10 cores for K3s and Arc |
| GPU | 1 × NVIDIA | Full GPU passthrough via nvidia runtime |
| Restart | unless-stopped | Auto-recover from OOM |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Ubuntu 24.04 base** (not Azure Linux) | `portaudio19-dev` unavailable on Azure Linux; avoids building from source |
| **GPU inference** (not CPU) | CPU inference pins all cores for 30-60s per utterance; GPU does it in 1-3s |
| **Disk-backed HF cache** (not tmpfs) | `/tmp` is tmpfs (RAM-backed) on Azure Linux — model stored in RAM twice if cached there |
| **Memory + CPU limits** | Without limits, NeMo + PyTorch can consume all RAM, crashing K3s and Arc SSH |

## Prerequisites

- Docker with NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- NVIDIA GPU driver loaded (`nvidia-smi` working)
- `/dev/snd` audio devices present

## Files

| File | Purpose |
|------|---------|
| `access_mic.py` | Main app — mic capture + NeMo STT |
| `Dockerfile` | Ubuntu 24.04 + PortAudio + NeMo |
| `docker-compose.yml` | GPU runtime + resource limits + cache volume |
| `requirements.txt` | Python dependencies |
| `Dockerfile.old` | Previous Azure Linux attempt (archived) |

## Session Changes Applied (April 24, 2026)

### Bug Fixes & Optimizations
| # | Change | File | Impact |
|---|--------|------|--------|
| 1 | **GPU auto-detect** — `.to("cpu")` → `torch.cuda` auto-detect | `access_mic.py` | Model uses GPU (1-3s inference vs 30-60s on CPU) |
| 2 | **Disk-backed HF cache** — `/tmp/hf-cache` → `/var/cache/hf-models` volume | `docker-compose.yml` | Eliminates 2.5GB RAM double-storage (tmpfs was RAM-backed) |
| 3 | **Memory limit 8GB** — unbounded → `memory: 8g` | `docker-compose.yml` | Prevents OOM-killing K3s and Arc agent |
| 4 | **CPU limit 4 cores** — unbounded → `cpus: 4` | `docker-compose.yml` | Leaves 10 cores for K3s, Arc, system services |
| 5 | **NVIDIA runtime** — missing → `runtime: nvidia` + GPU device reservation | `docker-compose.yml` | Proper GPU passthrough via nvidia-container-toolkit |
| 6 | **Restart policy** — none → `unless-stopped` | `docker-compose.yml` | Auto-recover from OOM kills |
| 7 | **NVIDIA env vars** — missing → `VISIBLE_DEVICES=0`, `CAPABILITIES=compute,utility` | `docker-compose.yml` | Correct GPU visibility inside container |

### Device Provisioning (on Lenovo SE100)
| # | Action | Persists? | Automated? |
|---|--------|:---------:|:----------:|
| 8 | Created `/dev/nvidia0` + `/dev/nvidiactl` device nodes | ❌ Reboot | ✅ `setup-device.sh` |
| 9 | Pruned 10 dead containers + 11 dangling images (13.8 GB freed) | ✅ | One-time |
| 10 | Deleted 15 failed K3s pods (post-reboot artifacts) | ✅ | ✅ `setup-device.sh` |
| 11 | Installed `alsa-utils` for audio diagnostics | ✅ | ✅ `setup-device.sh` |
| 12 | Installed Helm v3.20.2 | ✅ | ✅ `setup-device.sh` |
| 13 | Installed cert-manager v1.16.2 (3/3 pods running) | ✅ | ✅ `setup-device.sh` |
| 14 | Created `backupuser` (sudo, password auth) as safety net | ✅ | Manual |
| 15 | Fixed SSH key permissions (`icacls` on local `.pem`) | ✅ | One-time local |

### Not Yet Applied
| # | Item | Blocking? | Documented? |
|---|------|:---------:|:-----------:|
| 16 | `nvidia-container-toolkit` install | **Yes** (GPU in container) | ✅ `docs/DEVICE-SETUP.md` |
| 17 | `trust-manager` install | Yes (Foundry) | ✅ `docs/FOUNDRY-LOCAL.md` |
| 18 | Foundry Local install | Yes (model hosting) | ✅ `docs/FOUNDRY-LOCAL.md` |
| 19 | Audio buffer bounded deque | ⚠️ Memory leak over time | Planned |
| 20 | VS Code SSH config for Roycey | No | ✅ `docs/DEVICE-SETUP.md` |
| 21 | udev rule for persistent GPU nodes | ⚠️ Lost on reboot | ✅ `docs/DEVICE-SETUP.md` |

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/DEVICE-SETUP.md](docs/DEVICE-SETUP.md) | Device provisioning, SSH access, known issues |
| [docs/K3S-DEPLOYMENT.md](docs/K3S-DEPLOYMENT.md) | Running on K3s instead of Docker |
| [docs/FOUNDRY-LOCAL.md](docs/FOUNDRY-LOCAL.md) | Foundry Local analysis — why it doesn't fit ASR |
| [k8s/speech-recognizer.yaml](k8s/speech-recognizer.yaml) | K3s deployment manifest |
| [scripts/setup-device.sh](scripts/setup-device.sh) | Automated device setup (idempotent) |

## Credits

- Speech model: [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- Cobot POC: [Cosmos Darwin](https://github.com/cosmosdarwin/cobotpoc)
- Containerization: Roycey Cheeran
- GPU/resource optimization: Suhas Prakash
