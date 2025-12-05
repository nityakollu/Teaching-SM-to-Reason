# Distilled Models Directory

This directory contains fine-tuned LoRA adapters for small language models.

## Current Models

### `tiny-gpt2-lora/`
A LoRA adapter fine-tuned on tiny-gpt2 base model.

**Essential Files for Inference:**
- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - LoRA adapter weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.json` - Vocabulary file
- `merges.txt` - BPE merge rules
- `special_tokens_map.json` - Special token mappings

**Training Artifacts (Can Be Removed):**
- `checkpoint-5/` - Intermediate training checkpoint
  - Contains optimizer states, scheduler states, trainer state
  - Only needed if you want to resume training
  - Safe to delete if you're only doing inference

**Optional Files:**
- `README.md` - Auto-generated HuggingFace model card (can be removed)

## File Sizes

Current structure:
- Final adapter: ~4KB (adapter_config.json + adapter_model.safetensors)
- Tokenizer files: ~1.5MB (vocab.json + merges.txt + config files)
- Checkpoint directory: ~1.5MB (training artifacts)

**Total size**: ~3MB (can be reduced to ~1.5MB by removing checkpoint-5/)

## Usage

To use this model for inference:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "models/distilled/tiny-gpt2-lora")
```

Or use it in scripts:
```bash
python scripts/refine/refiner.py \
  --model models/distilled/tiny-gpt2-lora \
  --base-model sshleifer/tiny-gpt2
```

## Cleanup

To remove training artifacts and save space:

```bash
# Remove checkpoint directory (only if not resuming training)
rm -rf models/distilled/tiny-gpt2-lora/checkpoint-5/

# Remove auto-generated README (optional)
rm models/distilled/tiny-gpt2-lora/README.md
```

This reduces the model directory from ~3MB to ~1.5MB while keeping all files needed for inference.

