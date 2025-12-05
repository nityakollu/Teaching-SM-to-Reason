# Essential Files Guide - Teaching Small Models to Reason

## ✅ ESSENTIAL SCRIPTS (Keep These)

### Core Pipeline Scripts
1. **`scripts/run_pipeline.py`** - Main pipeline orchestrator (runs all steps)
2. **`scripts/dataset_generation/generate_examples.py`** - Generate datasets with Ollama
3. **`scripts/cleaning/filter_bad.py`** - Filter valid vs broken SMT programs
4. **`scripts/cleaning/repair_smt.py`** - Deterministic template-based repair
5. **`scripts/verify/check_validity.py`** - Z3 validation of SMT programs
6. **`scripts/distillation/format_jsonl.py`** - Convert CSV to JSONL format
7. **`scripts/distillation/finetune.py`** - LoRA fine-tuning script
8. **`scripts/refine/refiner.py`** - Solver-in-the-loop refinement
9. **`scripts/eval/evaluate.py`** - Final evaluation comparing baseline/distilled/refined

### Supporting Files
- **`requirements.txt`** - Python dependencies
- **`README.md`** - Project documentation
- **`QUICK_START.md`** - Quick start guide
- **`SETUP.md`** - Setup instructions

## ❌ DUPLICATE/UNUSED SCRIPTS (Can Delete)

### Dataset Generation Duplicates
- ❌ `scripts/dataset_generation/generate.py` - **DUPLICATE** (use `generate_examples.py` instead)

### Cleaning Scripts - Unused
- ❌ `scripts/cleaning/analyze_scheduling.py` - Analysis tool, not part of pipeline
- ❌ `scripts/cleaning/check_scheduling.py` - Analysis tool, not part of pipeline
- ❌ `scripts/cleaning/check_scheduling_simple.py` - Analysis tool, not part of pipeline
- ❌ `scripts/cleaning/clean_examples.py` - **DUPLICATE/UNUSED** (use `filter_bad.py` + `repair_smt.py`)

### Distillation Duplicates
- ❌ `scripts/distillation/finetune_lora.py` - **DUPLICATE** (use `finetune.py` instead)
- ❌ `scripts/distillation/format_for_training.py` - **DUPLICATE** (use `format_jsonl.py` instead)

### Refine Duplicates
- ❌ `scripts/refine/iterative_refiner.py` - **DUPLICATE** (use `refiner.py` instead)

### Evaluation Duplicates
- ❌ `scripts/eval/evaluate_models.py` - **DUPLICATE** (use `evaluate.py` instead)

### Other
- ❌ `scripts/generate_paper.py` - Optional, can be kept if needed

## 📁 ESSENTIAL DATA DIRECTORIES

### Input Data (Required)
```
data/raw/
  ├── scheduling.csv          # Required: Raw scheduling problems
  ├── puzzles.csv             # Required: Raw puzzle problems
  ├── relationships.csv       # Required: Raw relationship problems
  ├── resource_alloc.csv      # Required: Raw resource allocation problems
  └── broken.csv              # Generated: Broken examples from filtering
```

### Intermediate Data (Generated)
```
data/cleaned/
  ├── valid.csv               # Generated: Valid examples after filtering
  ├── repaired.csv            # Generated: Repaired examples
  └── training.jsonl          # Generated: Training data in JSONL format
```

### Evaluation Data (Generated)
```
data/eval/
  ├── validation_results.csv  # Generated: Z3 validation results
  ├── baseline_outputs.csv    # Generated: Baseline model outputs
  ├── distilled_outputs.csv   # Generated: Distilled model outputs
  ├── refined_outputs.csv     # Generated: Refined outputs
  └── final_metrics.json      # Generated: Final evaluation metrics
```

## 🗑️ DATA FILES TO CLEAN UP

### Duplicate/Old Data Files
- ❌ `data/cleaned/cleaned_dataset.csv` - **OLD/DEPRECATED** (use `valid.csv` + `repaired.csv` instead)
- ❌ `data/cleaned/jsonl/training_dataset.jsonl` - **DUPLICATE** (use `training.jsonl` instead)
- ❌ `data/eval/eval_set.csv` - **EMPTY/UNUSED** (remove if empty)
- ❌ `data/eval/model_outputs.csv` - **OLD TEST FILE** (remove)
- ❌ `data/eval/baseline_validation.csv` - **OLD** (replaced by `baseline_outputs.csv`)
- ❌ `data/eval/distilled_validation.csv` - **OLD** (replaced by `distilled_outputs.csv`)

### Optional Directories
- `data/raw/incomplete/` - Can keep for reference, not used in pipeline

## 📦 ESSENTIAL MODEL FILES

### Trained Models (LoRA Adapters)
```
models/distilled/
  └── <model-name>-lora/      # Generated: LoRA adapter after fine-tuning
      ├── adapter_config.json       # ✅ REQUIRED: LoRA configuration
      ├── adapter_model.safetensors # ✅ REQUIRED: LoRA weights
      ├── tokenizer_config.json     # ✅ REQUIRED: Tokenizer config
      ├── vocab.json                # ✅ REQUIRED: Vocabulary (~1MB)
      ├── merges.txt                # ✅ REQUIRED: BPE merges (~450KB)
      ├── special_tokens_map.json   # ✅ REQUIRED: Special tokens
      ├── checkpoint-5/             # ❌ OPTIONAL: Training checkpoint (can delete)
      └── README.md                 # ❌ OPTIONAL: Auto-generated (can delete)
```

**Essential Files for Inference:**
- `adapter_config.json` (~4KB)
- `adapter_model.safetensors` (~4KB)
- `tokenizer_config.json` (~4KB)
- `vocab.json` (~1MB)
- `merges.txt` (~450KB)
- `special_tokens_map.json` (~4KB)

**Can Be Removed (Training Artifacts):**
- `checkpoint-5/` directory (~1.5MB) - Only needed to resume training
  - Contains: optimizer.pt, scheduler.pt, trainer_state.json, rng_state.pth
  - Safe to delete if only doing inference
- `README.md` (~8KB) - Auto-generated by HuggingFace

**Total Size:** ~3MB (can reduce to ~1.5MB by removing checkpoint)

### Empty Directories (Can Create When Needed)
- `models/baseline/` - Empty, can create if storing baseline models
- `models/teacher/` - Empty, can create if storing teacher models
- `models/checkpoints/` - Empty, created during training

## 🚀 MINIMUM FILE SET TO RUN PIPELINE

To run the complete pipeline, you need:

### Scripts (9 files)
1. `scripts/run_pipeline.py`
2. `scripts/dataset_generation/generate_examples.py`
3. `scripts/cleaning/filter_bad.py`
4. `scripts/cleaning/repair_smt.py`
5. `scripts/verify/check_validity.py`
6. `scripts/distillation/format_jsonl.py`
7. `scripts/distillation/finetune.py`
8. `scripts/refine/refiner.py`
9. `scripts/eval/evaluate.py`

### Configuration
- `requirements.txt`
- `README.md`

### Input Data (or generate it)
- `data/raw/*.csv` (4 domain CSVs)

## 🔧 CLEANUP COMMANDS

To remove duplicate/unused files:

```bash
# Remove duplicate scripts
rm scripts/dataset_generation/generate.py
rm scripts/cleaning/clean_examples.py
rm scripts/distillation/finetune_lora.py
rm scripts/distillation/format_for_training.py
rm scripts/refine/iterative_refiner.py
rm scripts/eval/evaluate_models.py

# Remove analysis tools (optional)
rm scripts/cleaning/analyze_scheduling.py
rm scripts/cleaning/check_scheduling.py
rm scripts/cleaning/check_scheduling_simple.py

# Remove old data files
rm data/cleaned/cleaned_dataset.csv
rm data/cleaned/jsonl/training_dataset.jsonl
rm data/eval/model_outputs.csv
rm data/eval/baseline_validation.csv
rm data/eval/distilled_validation.csv

# Clean up model training artifacts (optional - only if not resuming training)
rm -rf models/distilled/tiny-gpt2-lora/checkpoint-5/
rm models/distilled/tiny-gpt2-lora/README.md  # Auto-generated, optional
```

## ✅ VERIFICATION CHECKLIST

Before running pipeline, verify:

- [ ] All 9 essential scripts exist
- [ ] `requirements.txt` is up to date
- [ ] Input CSVs exist in `data/raw/` OR generation script is ready
- [ ] Ollama is installed and accessible
- [ ] Python dependencies are installed (`pip install -r requirements.txt`)
- [ ] Z3 is installed (`pip install z3-solver`)
- [ ] HuggingFace transformers/peft are installed
- [ ] Model directory exists (`models/distilled/`)

## 📋 EXECUTION ORDER

1. **Generate datasets** → `data/raw/*.csv`
2. **Filter** → `data/cleaned/valid.csv` + `data/raw/broken.csv`
3. **Repair** → `data/cleaned/repaired.csv`
4. **Validate** → `data/eval/validation_results.csv`
5. **Format** → `data/cleaned/training.jsonl`
6. **Train** → `models/distilled/<model>-lora/`
7. **Refine** → `data/eval/refined_outputs.csv`
8. **Evaluate** → `data/eval/final_metrics.json`

All of this can be done with one command:
```bash
python scripts/run_pipeline.py
```

