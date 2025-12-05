# Teaching Small Models to Reason with SMT-LIB2

A complete end-to-end pipeline for teaching small language models to solve constraint satisfaction problems using SMT-LIB2, built entirely with local tools (Ollama + Python + Z3).

## Overview

This project implements a full pipeline for:

1. **Dataset Generation**: Using Ollama to generate constraint reasoning problems with SMT-LIB2 solutions
2. **Validation & Cleaning**: Z3-based validation and automatic filtering
3. **Fine-tuning**: LoRA-based fine-tuning of small models (Phi-3, Llama 3 8B, etc.)
4. **Iterative Refinement**: Solver-in-the-loop refinement for improving model outputs
5. **Evaluation**: Comprehensive evaluation comparing baseline, distilled, and refined models

## Quick Start

**Run the entire pipeline with one command:**

```bash
python scripts/run_pipeline.py
```

This automatically executes all 9 steps:
1. Dataset generation (300-400 samples)
2. Filtering bad outputs
3. Template-based SMT repair
4. Z3 validation
5. Format JSONL for training
6. LoRA fine-tuning
7. Iterative refinement
8. Evaluation (baseline + distilled + refined)
9. Generate paper draft

## Project Structure

```
teaching-small-models-reason/
├── data/
│   ├── raw/              # Raw generated examples (4 domain CSVs)
│   ├── cleaned/          # Validated and cleaned examples
│   └── eval/             # Evaluation outputs and metrics
├── models/
│   └── distilled/        # Fine-tuned LoRA adapters
├── scripts/
│   ├── dataset_generation/   # Example generation with Ollama
│   ├── cleaning/            # Filtering and template-based repair
│   ├── verify/              # Z3 validation
│   ├── distillation/        # Training pipeline
│   ├── refine/              # Solver-in-the-loop refinement
│   ├── eval/                # Model evaluation
│   └── run_pipeline.py      # Main pipeline orchestrator
├── README.md
├── SETUP.md
├── IMPLEMENTATION.md
├── ESSENTIAL_FILES.md
└── requirements.txt
```

## Key Features

- ✅ **100% Local**: No external APIs - uses Ollama, Python, and Z3
- ✅ **Deterministic Repair**: Template-based SMT repair (NO LLM)
- ✅ **Solver-in-the-Loop**: Z3-guided iterative refinement
- ✅ **Comprehensive Evaluation**: Compares baseline, distilled, and refined outputs
- ✅ **End-to-End Pipeline**: Single command execution

## Installation

See [SETUP.md](SETUP.md) for detailed installation instructions.

Quick setup:
```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama
ollama serve
ollama pull phi3
```

## Usage

### Full Pipeline

```bash
python scripts/run_pipeline.py
```

### Individual Steps

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed step-by-step instructions.

### Customization

```bash
# Skip generation (use existing data)
python scripts/run_pipeline.py --skip-generation

# Skip training (only evaluate)
python scripts/run_pipeline.py --skip-training

# Use different model
python scripts/run_pipeline.py --model llama3
```

## Pipeline Steps

1. **Dataset Generation** (`scripts/dataset_generation/generate_examples.py`)
   - Generates problems using Ollama
   - Outputs: `data/raw/*.csv`

2. **Filtering** (`scripts/cleaning/filter_bad.py`)
   - Detects valid vs broken SMT programs
   - Outputs: `data/cleaned/valid.csv`, `data/raw/broken.csv`

3. **Repair** (`scripts/cleaning/repair_smt.py`)
   - Deterministic template-based repair (NO LLM)
   - Outputs: `data/cleaned/repaired.csv`

4. **Validation** (`scripts/verify/check_validity.py`)
   - Z3 validation of all SMT programs
   - Outputs: `data/eval/validation_results.csv`

5. **Format** (`scripts/distillation/format_jsonl.py`)
   - Converts CSV to JSONL training format
   - Outputs: `data/cleaned/training.jsonl`

6. **Training** (`scripts/distillation/finetune.py`)
   - LoRA fine-tuning (r=16, alpha=16, dropout=0.05)
   - Outputs: `models/distilled/<model>-lora/`

7. **Refinement** (`scripts/refine/refiner.py`)
   - Solver-in-the-loop iterative refinement
   - Outputs: `data/eval/refined_outputs.csv`

8. **Evaluation** (`scripts/eval/evaluate.py`)
   - Compares baseline, distilled, and refined models
   - Outputs: `data/eval/final_metrics.json`

## Requirements

- Python 3.8+
- Ollama installed and running
- Z3 solver (`pip install z3-solver`)
- PyTorch, Transformers, PEFT libraries
- CUDA-capable GPU (recommended for training)

## Documentation

- **[SETUP.md](SETUP.md)** - Detailed setup and installation guide
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Implementation details and architecture
- **[ESSENTIAL_FILES.md](ESSENTIAL_FILES.md)** - File structure and cleanup guide

## License

See LICENSE file for details.

## Citation

If you use this project, please cite:

```bibtex
@software{teaching_small_models_reason,
  title = {Teaching Small Models to Reason with SMT-LIB2},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/teaching-small-models-reason}
}
```
