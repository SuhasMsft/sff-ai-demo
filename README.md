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

## Credits

- Speech model: [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- Cobot POC: [Cosmos Darwin](https://github.com/cosmosdarwin/cobotpoc)
- Containerization: Roycey Cheeran
- GPU/resource optimization: Suhas Prakash
