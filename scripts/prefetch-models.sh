#!/bin/bash
# prefetch-models.sh — Download all model weights to persistent disk cache
# Run BEFORE starting containers to avoid runtime download delays
set -euo pipefail

CACHE_DIR="${HF_HOME:-/acsa/hf-models}"
echo "=== Pre-fetching models to $CACHE_DIR ==="

sudo mkdir -p "$CACHE_DIR"
sudo chmod 777 "$CACHE_DIR"
export HF_HOME="$CACHE_DIR"

# Model 1: NeMo Parakeet TDT 0.6B v2 (ASR)
echo "[1/3] Downloading Parakeet TDT 0.6B v2..."
python3 -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v2')
print('Parakeet downloaded successfully')
del model
" 2>&1 | tail -5

# Model 2: Grounding-DINO-Base (Vision)
echo "[2/3] Downloading Grounding-DINO-Base..."
python3 -c "
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
proc = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-base')
model = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-base')
print('Grounding-DINO downloaded successfully')
del model, proc
" 2>&1 | tail -5

# Model 3: Qwen 2.5 0.5B Instruct (LLM)
echo "[3/3] Downloading Qwen 2.5 0.5B Instruct..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('Qwen downloaded successfully')
del model, tok
" 2>&1 | tail -5

echo ""
echo "=== All models cached at $CACHE_DIR ==="
du -sh "$CACHE_DIR"
