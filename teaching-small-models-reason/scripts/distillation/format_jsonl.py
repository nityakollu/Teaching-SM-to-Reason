"""
Convert cleaned CSV data into JSONL training format.

Usage:
    python scripts/distillation/format_jsonl.py \
        --input data/cleaned/repaired.csv \
        --output data/cleaned/training.jsonl

Each JSONL entry:
{
  "instruction": "Convert the constraint problem into reasoning and SMT-LIB.",
  "input": "<natural language problem>",
  "output": "<reasoning>\n\n<SMT program>"
}
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "cleaned" / "repaired.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "cleaned" / "training.jsonl"
INSTRUCTION_TEXT = "Convert the constraint problem into reasoning and SMT-LIB."


def iter_rows(csv_path: Path) -> Iterator[Dict[str, str]]:
    """Yield normalized rows from the CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        # Detect delimiter by inspecting the first line
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue

            normalized = {
                key.strip().lower(): (value or "").strip()
                for key, value in row.items()
                if key is not None
            }

            problem = normalized.get("problem") or normalized.get("prompt") or ""
            reasoning = normalized.get("reasoning") or normalized.get("explanation") or ""
            program = normalized.get("program") or normalized.get("smt_program") or ""

            if not problem and not reasoning and not program:
                continue

            yield {
                "problem": problem,
                "reasoning": reasoning,
                "program": program,
            }


def convert_to_jsonl(input_path: Path, output_path: Path) -> None:
    """Convert CSV rows into the expected JSONL training format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for row in iter_rows(input_path):
            total += 1
            output_record = {
                "instruction": INSTRUCTION_TEXT,
                "input": row["problem"],
                "output": f"{row['reasoning']}\n\n{row['program']}".strip(),
            }
            out_f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {total} training examples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cleaned CSV data into JSONL training format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Input CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Make paths relative to project root when given as relative strings
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    convert_to_jsonl(input_path, output_path)


if __name__ == "__main__":
    main()
"""
Create JSONL training data from cleaned CSV.

Format: {"instruction": "...", "input": "<problem>", "output": "Reasoning: ...\nProgram:\n<program>"}
"""

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
CLEANED_DATASET = CLEANED_DIR / "cleaned_dataset.csv"
TRAINING_JSONL = CLEANED_DIR / "training.jsonl"


def format_example(problem: str, reasoning: str, program: str) -> dict:
    """
    Format a single example into instruction/input/output format.
    
    Args:
        problem: Problem statement
        reasoning: Step-by-step reasoning
        program: SMT-LIB2 program
    
    Returns:
        Dictionary with instruction, input, and output
    """
    instruction = """You are an expert in constraint programming and SMT-LIB2. 
Given a problem description, provide step-by-step reasoning and generate a valid SMT-LIB2 program to solve it."""
    
    input_text = problem.strip()
    
    output_text = f"""Reasoning:
{reasoning}

Program:
{program}"""
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def convert_csv_to_jsonl(input_csv: Path, output_jsonl: Path) -> int:
    """
    Convert cleaned CSV to JSONL format.
    
    Args:
        input_csv: Path to input CSV file
        output_jsonl: Path to output JSONL file
    
    Returns:
        Number of examples converted
    """
    examples = []
    
    if not input_csv.exists():
        print(f"Error: Input file not found: {input_csv}")
        return 0
    
    # Read CSV
    with input_csv.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.reader(f, delimiter=delimiter)
        
        for row in reader:
            if not row or len(row) < 3:
                continue
            
            problem = row[0].strip()
            reasoning = row[1].strip()
            program = row[2].strip()
            
            if problem and reasoning and program:
                formatted = format_example(problem, reasoning, program)
                examples.append(formatted)
    
    # Write JSONL
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    return len(examples)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CSV to JSONL training format")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help=f"Input CSV file (default: {CLEANED_DATASET})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help=f"Output JSONL file (default: {TRAINING_JSONL})",
    )
    
    args = parser.parse_args()
    
    input_csv = Path(args.input) if args.input else CLEANED_DATASET
    output_jsonl = Path(args.output) if args.output else TRAINING_JSONL
    
    print("="*60)
    print("JSONL Format Conversion")
    print("="*60)
    print(f"Input: {input_csv}")
    print(f"Output: {output_jsonl}")
    print()
    
    count = convert_csv_to_jsonl(input_csv, output_jsonl)
    
    print(f"✓ Converted {count} examples to JSONL format")
    print(f"  Saved to: {output_jsonl}")


if __name__ == "__main__":
    main()

