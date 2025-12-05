# Implementation Details

Complete technical documentation for the Teaching Small Models to Reason pipeline.

## Architecture Overview

The pipeline consists of 9 sequential steps, each implemented as a standalone Python script that can be run independently or orchestrated via `scripts/run_pipeline.py`.

## Pipeline Steps

### Step 1: Dataset Generation

**Script**: `scripts/dataset_generation/generate_examples.py`

**Purpose**: Generate constraint reasoning problems using Ollama.

**Process**:
- Connects to local Ollama instance
- Generates problems across 4 domains: scheduling, puzzles, relationships, resource allocation
- Each problem includes: natural language description, reasoning steps, SMT-LIB2 program
- Uses batch processing (20 examples per batch) for efficiency

**Output**:
- `data/raw/scheduling.csv`
- `data/raw/puzzles.csv`
- `data/raw/relationships.csv`
- `data/raw/resource_alloc.csv`

**Usage**:
```bash
python scripts/dataset_generation/generate_examples.py
```

**Parameters**:
- `--model`: Ollama model name (default: phi3)
- `--num-examples`: Examples per domain (default: 100)
- `--batch-size`: Batch size for generation (default: 20)

---

### Step 2: Filtering

**Script**: `scripts/cleaning/filter_bad.py`

**Purpose**: Separate valid SMT programs from broken ones.

**Process**:
- Reads all domain CSVs from `data/raw/`
- Detects valid SMT programs (proper syntax, balanced parentheses, required components)
- Detects broken programs (missing parentheses, syntax errors, empty)
- Uses regex patterns and structural checks

**Output**:
- `data/cleaned/valid.csv` - All valid examples
- `data/raw/broken.csv` - All broken examples with error details

**Usage**:
```bash
python scripts/cleaning/filter_bad.py
```

**Validation Criteria**:
- Has `declare-const` or `declare-fun`
- Has `assert` statement
- Balanced parentheses
- Minimum length (20 characters)
- No severe syntax errors

---

### Step 3: Template-Based Repair

**Script**: `scripts/cleaning/repair_smt.py`

**Purpose**: Deterministically repair broken SMT programs using templates (NO LLM).

**Process**:
- Reads broken examples from `data/raw/broken.csv`
- Parses natural language problem to extract:
  - Variables (C1, A, task T1, etc.)
  - Numbers (for bounds, slots, values)
  - Constraint types (friend, scheduling, equality, etc.)
  - Domain (scheduling, relationships, puzzles, resource allocation)
- Applies appropriate template based on detected pattern
- Generates exactly 1 repaired row per input row

**Templates Available**:
- Boolean constraints
- Integer constraints
- Equality/inequality
- Friend relationships
- Scheduling constraints
- Integer arithmetic

**Output**:
- `data/cleaned/repaired.csv` - All repaired examples (same row count as broken.csv)

**Usage**:
```bash
python scripts/cleaning/repair_smt.py
```

**Key Features**:
- ✅ Fully deterministic (NO LLM calls)
- ✅ One-to-one mapping (1 input row → 1 output row)
- ✅ All outputs are syntactically valid SMT-LIB2

---

### Step 4: Z3 Validation

**Script**: `scripts/verify/check_validity.py`

**Purpose**: Validate all SMT programs using Z3 theorem prover.

**Process**:
- Reads CSV file with SMT programs
- For each program:
  - Parses with Z3's `parse_smt2_string()`
  - Adds assertions to solver
  - Checks satisfiability
- Records validation status and errors

**Output**:
- `data/eval/validation_results.csv` - Validation results (id, valid, error)

**Usage**:
```bash
python scripts/verify/check_validity.py data/cleaned/repaired.csv --output data/eval/validation_results.csv
```

**Validation Results**:
- `valid=True`: Program parsed successfully and can be checked
- `valid=False`: Parse error or validation failure
- `error`: Error message if invalid

---

### Step 5: Format JSONL

**Script**: `scripts/distillation/format_jsonl.py`

**Purpose**: Convert CSV data to JSONL format for training.

**Process**:
- Reads CSV with columns: problem, reasoning, program
- Formats each row as JSONL entry:
  ```json
  {
    "instruction": "Convert the constraint problem into reasoning and SMT-LIB.",
    "input": "<natural language problem>",
    "output": "<reasoning>\n\n<SMT program>"
  }
  ```

**Output**:
- `data/cleaned/training.jsonl` - Training data in JSONL format

**Usage**:
```bash
python scripts/distillation/format_jsonl.py --input data/cleaned/repaired.csv --output data/cleaned/training.jsonl
```

---

### Step 6: LoRA Fine-tuning

**Script**: `scripts/distillation/finetune.py`

**Purpose**: Fine-tune a small language model using LoRA.

**Process**:
- Loads base model (phi-3, llama-3-8b, etc.)
- Configures LoRA with:
  - `r=16` (rank)
  - `alpha=16` (scaling)
  - `dropout=0.05`
- Trains on JSONL data
- Uses 4-bit quantization for memory efficiency

**Output**:
- `models/distilled/<model-name>-lora/` - LoRA adapter files

**Usage**:
```bash
python scripts/distillation/finetune.py \
  --model microsoft/Phi-2 \
  --data data/cleaned/training.jsonl \
  --output models/distilled/phi2-lora \
  --max-steps 1000 \
  --batch-size 2
```

**Parameters**:
- `--model`: Base model name or path
- `--data`: Training JSONL file
- `--output`: Output directory for adapter
- `--max-steps`: Training steps
- `--batch-size`: Batch size
- `--lr`: Learning rate (default: 2e-4)

---

### Step 7: Solver-in-the-Loop Refinement

**Script**: `scripts/refine/refiner.py`

**Purpose**: Iteratively refine model outputs using Z3 feedback.

**Process**:
- Reads dataset (valid + repaired examples)
- For each example:
  1. Validate initial SMT program with Z3
  2. If invalid, extract Z3 error message
  3. Ask distilled model to fix ONLY the reported error
  4. Iterate up to 3 times
  5. Record initial validity, final validity, and iterations

**Output**:
- `data/eval/refined_outputs.csv` - Refinement statistics (id, initial_valid, final_valid, iterations)

**Usage**:
```bash
python scripts/refine/refiner.py \
  --output data/eval/refined_outputs.csv \
  --model models/distilled/tiny-gpt2-lora \
  --base-model sshleifer/tiny-gpt2 \
  --max-iters 3
```

**Key Features**:
- Uses Z3 to detect errors
- Targeted correction prompts
- Maximum 3 refinement iterations
- Processes all examples in dataset

---

### Step 8: Evaluation

**Script**: `scripts/eval/evaluate.py`

**Purpose**: Compare baseline, distilled, and refined model outputs.

**Process**:
- Loads evaluation dataset
- For each example:
  1. **Baseline**: Generate SMT using Ollama model
  2. **Distilled**: Generate SMT using fine-tuned model
  3. **Refined**: Run solver-in-the-loop refinement on distilled output
- Validates all outputs with Z3
- Computes validity percentages

**Output**:
- `data/eval/baseline_outputs.csv`
- `data/eval/distilled_outputs.csv`
- `data/eval/refined_outputs.csv`
- `data/eval/final_metrics.json` - Summary metrics

**Usage**:
```bash
python scripts/eval/evaluate.py \
  --dataset data/cleaned/repaired.csv \
  --output-dir data/eval \
  --baseline-model phi3 \
  --distilled-model models/distilled/tiny-gpt2-lora \
  --distilled-base sshleifer/tiny-gpt2 \
  --max-examples 100
```

**Metrics Computed**:
- `baseline_valid_pct`: % of baseline outputs that are valid
- `distilled_valid_pct`: % of distilled outputs that are valid
- `refined_valid_pct`: % of refined outputs that are valid

---

## Key Design Decisions

### 1. Deterministic Repair (NO LLM)

The repair step uses template matching rather than LLM generation:
- ✅ Reproducible results
- ✅ Fast execution
- ✅ Guaranteed syntactic validity
- ✅ No API costs

### 2. Solver-in-the-Loop

Refinement uses Z3 feedback to guide corrections:
- ✅ Targeted error fixing
- ✅ Iterative improvement
- ✅ Maximum 3 iterations to prevent loops

### 3. End-to-End Pipeline

All steps can run independently or via orchestrator:
- ✅ Flexible execution
- ✅ Easy debugging
- ✅ Parallel development

### 4. Local-Only Tools

Everything runs locally:
- ✅ No external API dependencies
- ✅ Privacy-preserving
- ✅ Offline-capable
- ✅ Cost-effective

## File Formats

### CSV Format (Input/Output)

```csv
problem,reasoning,program
"Problem description", "Reasoning steps", "(set-logic QF_LIA)\n(declare-const x Int)\n..."
```

### JSONL Format (Training)

```json
{"instruction": "Convert the constraint problem into reasoning and SMT-LIB.", "input": "Problem description", "output": "Reasoning steps\n\n(set-logic QF_LIA)\n..."}
```

### Metrics JSON (Output)

```json
{
  "baseline_valid_pct": 75.5,
  "distilled_valid_pct": 82.3,
  "refined_valid_pct": 91.2,
  "num_examples": 100,
  "total_runtime_seconds": 1234.5
}
```

## Performance Considerations

### Dataset Generation
- Batch processing (20 examples per batch)
- Ollama connection pooling
- ~5-10 minutes for 400 examples

### Training
- 4-bit quantization reduces memory by ~75%
- LoRA reduces trainable parameters by ~99%
- ~1-2 hours on GPU for 400 examples

### Evaluation
- Z3 validation is fast (<1s per program)
- Model inference is the bottleneck
- ~10-30 minutes for 100 examples

## Troubleshooting

See [SETUP.md](SETUP.md) for common issues and solutions.

## Extending the Pipeline

### Adding New Domains

1. Add domain to `generate_examples.py`
2. Update filtering logic if needed
3. Add domain-specific templates to `repair_smt.py`

### Adding New Templates

1. Add template method to `SMTTemplateLibrary` class
2. Update parser to detect new constraint type
3. Add template selection logic to `SMTRepairer.repair()`

### Changing Models

1. Update model paths in scripts
2. Adjust batch size and sequence length
3. Tune learning rate if needed

## References

- SMT-LIB2 specification: http://smtlib.cs.uiowa.edu/
- Z3 documentation: https://github.com/Z3Prover/z3
- PEFT documentation: https://huggingface.co/docs/peft
- Ollama documentation: https://github.com/ollama/ollama

