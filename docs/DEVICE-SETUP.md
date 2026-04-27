# Device Setup Guide

All changes needed to prepare an Azure Local SFF device for the speech demo.

## What Survives What

| Change | Survives Reboot? | Survives A/B Swap? | Automated? |
|--------|:---:|:---:|:---:|
| GPU device nodes (`/dev/nvidia0`) | ❌ | ❌ | ✅ `setup-device.sh` |
| alsa-utils package | ✅ | ❌ | ✅ `setup-device.sh` |
| Helm binary | ✅ | ❌ | ✅ `setup-device.sh` |
| cert-manager (K3s) | ✅ | ❌ | ✅ `setup-device.sh` |
| trust-manager (K3s) | ✅ | ❌ | ✅ `setup-device.sh` |
| HF cache directory | ✅ | ❌ | ✅ `setup-device.sh` |
| Docker images | ✅ | ❌ | Manual rebuild |
| K3s pod cleanup | ⚠️ Temporary | N/A | ✅ `setup-device.sh` |

## Quick Start

```bash
# Run the setup script (idempotent — safe to re-run)
chmod +x scripts/setup-device.sh
./scripts/setup-device.sh
```

## Manual Steps (not in script)

### GPU Device Nodes on Every Reboot

`/dev` is tmpfs on Azure Linux — GPU nodes vanish on reboot. Add a udev rule for persistence:

```bash
# Create persistent udev rule
echo 'ACTION=="add", DEVPATH=="/module/nvidia", RUN+="/usr/bin/mknod -m 666 /dev/nvidia0 c 195 0"' | \
    sudo tee /etc/udev/rules.d/99-nvidia.rules
echo 'ACTION=="add", DEVPATH=="/module/nvidia", RUN+="/usr/bin/mknod -m 666 /dev/nvidiactl c 195 255"' | \
    sudo tee -a /etc/udev/rules.d/99-nvidia.rules
sudo udevadm control --reload-rules
```

### NVIDIA Container Toolkit (for `--runtime=nvidia`)

**Not yet installed.** Required for GPU passthrough in Docker containers:

```bash
# Azure Linux 3.0 — install from NVIDIA repo
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo tee /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo tdnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Backup User (optional safety net)

```bash
sudo useradd -m -G sudo -s /bin/bash backupuser
sudo passwd backupuser
# Verify: id backupuser
# Test SSH in second terminal before closing current session
```

## SSH Access

```bash
az ssh arc \
  --subscription 8d7607ea-9060-4cb6-a3c9-e562b14f9d14 \
  --resource-group AzLocal-ManagedResources-rcheeran-build-14e2 \
  --name 651441a2-1e90-4e87-afd0-6c54402789df-1 \
  --local-user clouduser \
  --private-key-file ~/.ssh/lenovo-demo-ssh.pem \
  -- -o StrictHostKeyChecking=no
```

## Known Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `nvidia-smi` fails after reboot | `/dev/nvidia0` missing (tmpfs) | Run `setup-device.sh` or add udev rule |
| Arc SSH "404 timeout" | Device under heavy load (model download/inference) | Wait or reboot; apply memory limits |
| `himds.service not found` | Azure Linux uses `azcmagent`, not `himds` | Use `sudo azcmagent show` instead |
| Device "very unresponsive" | NeMo model cached in tmpfs (RAM) + CPU inference | Use disk-backed HF_HOME + GPU inference |
