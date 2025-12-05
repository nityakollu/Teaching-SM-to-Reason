"""
Solver-in-the-loop refinement for SMT programs.

Workflow:
1. Read raw model outputs from a CSV file (default: data/eval/model_outputs.csv).
2. Validate each SMT program with Z3.
3. If invalid, extract solver errors and ask the distilled model to fix ONLY that issue.
4. Iterate up to --max-iters times.
5. Write refinement statistics to data/eval/refined_outputs.csv with columns:
   id, initial_valid, final_valid, iterations.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple

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
DEFAULT_INPUT = PROJECT_ROOT / "data" / "cleaned" / "repaired.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "refined_outputs.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "distilled" / "phi2-lora"


def validate_program(program: str) -> Tuple[bool, Optional[str]]:
    """Validate SMT program with Z3. Returns (is_valid, error_message)."""
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


def build_refinement_prompt(problem: str, current_program: str, error: str) -> str:
    """Create a targeted correction prompt."""
    return (
        "You are an expert SMT-LIB compiler. "
        "Fix ONLY the error reported by Z3 and return a corrected SMT-LIB program. "
        "Do not add explanations, comments, or prose.\n\n"
        f"Problem:\n{problem.strip() or 'N/A'}\n\n"
        "Current SMT-LIB program:\n"
        f"{current_program.strip()}\n\n"
        "Z3 error message:\n"
        f"{error.strip()}\n\n"
        "Corrected SMT-LIB program:\n"
    )


def extract_program_from_response(response: str) -> str:
    """Extract SMT program from model response (handles code fences)."""
    response = response.strip()
    if not response:
        return ""

    fence_match = re.search(r"```(?:smt2|lisp|)\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    return response


def load_model(model_path: Path, base_model_name: Optional[str] = None):
    """
    Load tokenizer and model (with optional PEFT adapter).
    """
    model_path = Path(model_path)
    if base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(model, model_path)
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model


def generate_fix(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a refinement using the distilled model."""
    device = model.device if hasattr(model, "device") else torch.device("cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    return decoded.strip()


def read_input_rows(csv_path: Path):
    """Yield rows from CSV with normalized keys."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            if not row:
                continue

            problem = (row.get("problem") or row.get("Problem") or "").strip()
            reasoning = (row.get("reasoning") or row.get("Reasoning") or "").strip()
            program = (row.get("program") or row.get("Program") or "").strip()

            if not problem:
                continue

            yield {
                "problem": problem,
                "program": program,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solver-in-the-loop SMT refiner.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"CSV file with model outputs (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV for refinement stats (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help=f"Local HF model or LoRA adapter path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional base model name if --model points to a PEFT adapter",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum refinement iterations per example (default: 3)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens during refinement generation (default: 512)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_path = Path(args.model)

    tokenizer, model = load_model(model_path, args.base_model)

    output_rows = []

    for idx, row in enumerate(read_input_rows(input_path), start=1):
        problem = row["problem"]
        candidate_program = row["program"]

        initial_valid, error = validate_program(candidate_program)
        final_valid = initial_valid
        iterations = 0

        while (not final_valid) and iterations < args.max_iters:
            iterations += 1
            prompt = build_refinement_prompt(problem, candidate_program, error or "Unknown error")
            response = generate_fix(
                tokenizer,
                model,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            candidate_program = extract_program_from_response(response)
            final_valid, error = validate_program(candidate_program)

        output_rows.append(
            {
                "id": idx,
                "initial_valid": initial_valid,
                "final_valid": final_valid,
                "iterations": iterations,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "initial_valid", "final_valid", "iterations"],
        )
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    total = len(output_rows)
    improved = sum(1 for r in output_rows if (not r["initial_valid"]) and r["final_valid"])
    print(f"✓ Processed {total} examples. Improved {improved} programs.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

