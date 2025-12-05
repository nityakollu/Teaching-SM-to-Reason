"""
Evaluate baseline, distilled, and solver-refined models on SMT problems.

The script:
1. Loads an evaluation dataset (problem, reasoning, reference program).
2. Generates SMT programs with:
   - Baseline model (Ollama)
   - Distilled local model (HuggingFace checkpoint or LoRA adapter)
3. Runs solver-in-the-loop refinement on the distilled outputs.
4. Validates every generated SMT program with Z3.
5. Saves per-example outputs and aggregates final metrics.

Usage:
    python scripts/eval/evaluate.py \
        --dataset data/cleaned/repaired.csv \
        --output-dir data/eval \
        --baseline-model phi3 \
        --distilled-model models/distilled/tiny-gpt2-lora \
        --distilled-base sshleifer/tiny-gpt2 \
        --max-examples 100
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from z3 import Solver, parse_smt2_string
except ImportError as exc:
    raise SystemExit(
        "Error: z3-solver not installed. Install with `pip install z3-solver`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "cleaned" / "repaired.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"
DEFAULT_BASELINE_MODEL = "phi3"
DEFAULT_DISTILLED_MODEL = PROJECT_ROOT / "models" / "distilled" / "phi2-lora"
DEFAULT_DISTILLED_BASE = "microsoft/phi-2"

BASELINE_PROMPT = """You are an SMT solver assistant.
Given the following problem, output ONLY a complete SMT-LIB2 program that can be checked with Z3.
Do not include explanations or comments.

Problem:
{problem}
"""

DISTILLED_PROMPT = """[INSTRUCTION] Convert the constraint problem into reasoning and SMT-LIB.
[INPUT] {problem}
[OUTPUT]"""

REFINEMENT_PROMPT = """You are an expert SMT-LIB compiler. Fix ONLY the error reported by Z3.

Problem:
{problem}

Current SMT-LIB program:
{program}

Z3 error:
{error}

Return ONLY the corrected SMT-LIB program without explanations."""


def load_dataset(path: Path, limit: Optional[int] = None) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            problem = (row.get("problem") or row.get("Problem") or "").strip()
            reasoning = (row.get("reasoning") or row.get("Reasoning") or "").strip()
            program = (row.get("program") or row.get("Program") or "").strip()
            if not problem:
                continue
            rows.append({"problem": problem, "reasoning": reasoning, "reference": program})
            if limit and len(rows) >= limit:
                break
    return rows


def validate_program(program: str) -> Tuple[bool, Optional[str]]:
    program = (program or "").strip()
    if not program:
        return False, "Empty SMT program"
    try:
        solver = Solver()
        assertions = parse_smt2_string(program)
        for assertion in assertions:
            solver.add(assertion)
        solver.check()
        return True, None
    except Exception as exc:
        return False, str(exc)


def extract_program(text: str) -> str:
        import re
        
    text = text.strip()
    if not text:
        return ""

    match = re.search(r"```(?:smt2|smt|lisp)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    if "(declare-const" in text and "(check-sat" in text:
            start = text.find("(declare-const")
        end = text.rfind("(check-sat")
        if end != -1:
            closing = text[end:]
            return text[start : end + len(closing.splitlines()[0])].strip()
    return text


def run_ollama(model_name: str, prompt: str, timeout: int = 120) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Unknown Ollama error")
        return result.stdout.strip()
    except FileNotFoundError as exc:
        raise RuntimeError("Ollama CLI not found on PATH") from exc


def load_distilled_model(model_path: Path, base_model: str):
    if model_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()
    return tokenizer, model


def generate_with_distilled(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        max_length=512,
    ).to(device)
        with torch.no_grad():
        outputs = model.generate(
                **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
                do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    return text


def refine_program(problem: str, program: str, tokenizer, model, max_iters: int, max_new_tokens: int):
    candidate = program
    valid, error = validate_program(candidate)
        iterations = 0
        
    while not valid and iterations < max_iters:
        iterations += 1
        prompt = REFINEMENT_PROMPT.format(problem=problem, program=candidate, error=error or "Unknown error")
        response = generate_with_distilled(tokenizer, model, prompt, max_new_tokens)
        candidate = extract_program(response)
        valid, error = validate_program(candidate)

    return candidate, valid, iterations


def evaluate_dataset(
    dataset: List[Dict[str, str]],
    baseline_model: str,
    tokenizer,
    distilled_model,
    max_new_tokens: int,
    max_refine_iters: int,
    output_dir: Path,
):
    baseline_rows = []
    distilled_rows = []
    refined_rows = []

    for idx, example in enumerate(dataset, start=1):
        problem = example["problem"]

        # Baseline via Ollama
        try:
            baseline_raw = run_ollama(baseline_model, BASELINE_PROMPT.format(problem=problem))
            baseline_program = extract_program(baseline_raw)
            baseline_valid, baseline_error = validate_program(baseline_program)
        except Exception as exc:
            baseline_program = ""
            baseline_valid = False
            baseline_error = str(exc)
        baseline_rows.append(
            {
                "id": idx,
                "problem": problem,
                "program": baseline_program,
                "valid": baseline_valid,
                "error": baseline_error or "",
            }
        )

        # Distilled model
        try:
            distilled_raw = generate_with_distilled(
                tokenizer,
                distilled_model,
                DISTILLED_PROMPT.format(problem=problem),
                max_new_tokens=max_new_tokens,
            )
            distilled_program = extract_program(distilled_raw)
            distilled_valid, distilled_error = validate_program(distilled_program)
        except Exception as exc:
            distilled_program = ""
            distilled_valid = False
            distilled_error = str(exc)
        distilled_rows.append(
            {
                "id": idx,
                "problem": problem,
                "program": distilled_program,
                "valid": distilled_valid,
                "error": distilled_error or "",
            }
        )

        # Solver-in-the-loop refinement
        refined_program, refined_valid, iterations = refine_program(
            problem,
            distilled_program,
            tokenizer,
            distilled_model,
            max_refine_iters,
            max_new_tokens,
        )
        refined_rows.append(
            {
                "id": idx,
                "problem": problem,
                "program": refined_program,
                "initial_valid": distilled_valid,
                "final_valid": refined_valid,
                "iterations": iterations,
            }
        )

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(dataset)} examples...")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(output_dir / "baseline_outputs.csv", ["id", "problem", "program", "valid", "error"], baseline_rows)
    save_csv(output_dir / "distilled_outputs.csv", ["id", "problem", "program", "valid", "error"], distilled_rows)
    save_csv(
        output_dir / "refined_outputs.csv",
        ["id", "problem", "program", "initial_valid", "final_valid", "iterations"],
        refined_rows,
    )

    return baseline_rows, distilled_rows, refined_rows


def save_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: List[Dict], valid_key: str) -> float:
    if not rows:
        return 0.0
    total = len(rows)
    valid = sum(1 for row in rows if row.get(valid_key))
    return (valid / total) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline, distilled, and refined models on SMT tasks.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help=f"CSV dataset with columns problem,reasoning,program (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to store outputs and metrics (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=DEFAULT_BASELINE_MODEL,
        help="Ollama model name for baseline (default: phi3)",
    )
    parser.add_argument(
        "--distilled-model",
        type=str,
        default=str(DEFAULT_DISTILLED_MODEL),
        help=f"HuggingFace model or PEFT adapter path (default: {DEFAULT_DISTILLED_MODEL})",
    )
    parser.add_argument(
        "--distilled-base",
        type=str,
        default=DEFAULT_DISTILLED_BASE,
        help=f"Base model name if --distilled-model is a LoRA adapter (default: {DEFAULT_DISTILLED_BASE})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for each model output (default: 512)",
    )
    parser.add_argument(
        "--max-refine-iters",
        type=int,
        default=3,
        help="Maximum solver-in-loop refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit on number of evaluation examples",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="final_metrics.json",
        help="Filename for metrics JSON inside output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    examples = load_dataset(dataset_path, args.max_examples)
    print(f"Loaded {len(examples)} examples from {dataset_path}")

    tokenizer, distilled_model = load_distilled_model(Path(args.distilled_model), args.distilled_base)

    start = time.time()
    baseline_rows, distilled_rows, refined_rows = evaluate_dataset(
        examples,
        args.baseline_model,
        tokenizer,
        distilled_model,
        args.max_new_tokens,
        args.max_refine_iters,
        output_dir,
    )
    elapsed = time.time() - start

    metrics = {
        "baseline_valid_pct": summarize(baseline_rows, "valid"),
        "distilled_valid_pct": summarize(distilled_rows, "valid"),
        "refined_valid_pct": summarize(refined_rows, "final_valid"),
        "examples": len(examples),
        "elapsed_sec": elapsed,
    }

    metrics_path = output_dir / args.metrics_file
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nEvaluation complete.")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()

