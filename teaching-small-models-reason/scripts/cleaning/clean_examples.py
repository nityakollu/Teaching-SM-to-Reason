"""
Filter and clean examples using Z3 validation.

Removes invalid SMT programs and keeps only valid, satisfiable examples.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify.check_validity import SMTValidator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"


def clean_csv_file(
    input_path: Path,
    output_path: Path,
    require_sat: bool = True,
    min_problem_length: int = 20,
    min_reasoning_length: int = 50,
) -> Dict[str, int]:
    """
    Clean examples from a CSV file.
    
    Args:
        input_path: Input CSV file path
        output_path: Output CSV file path
        require_sat: Only keep satisfiable examples
        min_problem_length: Minimum problem statement length
        min_reasoning_length: Minimum reasoning length
    
    Returns:
        Dictionary with cleaning statistics
    """
    stats = {
        "total": 0,
        "valid": 0,
        "invalid_syntax": 0,
        "unsat_or_unknown": 0,
        "too_short": 0,
        "kept": 0,
    }
    
    validator = SMTValidator()
    cleaned_examples = []
    
    # Read input file
    with input_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.reader(f, delimiter=delimiter)
        
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            
            if len(row) < 3:
                continue
            
            problem = row[0].strip()
            reasoning = row[1].strip()
            program = row[2].strip()
            
            stats["total"] += 1
            
            # Check length requirements
            if len(problem) < min_problem_length or len(reasoning) < min_reasoning_length:
                stats["too_short"] += 1
                continue
            
            # Validate SMT program
            is_valid, status, metadata = validator.validate_program(program)
            
            if not is_valid:
                stats["invalid_syntax"] += 1
                continue
            
            stats["valid"] += 1
            
            # Check satisfiability requirement
            if require_sat and status != "sat":
                stats["unsat_or_unknown"] += 1
                continue
            
            # Keep this example
            cleaned_examples.append([problem, reasoning, program])
            stats["kept"] += 1
    
    # Write cleaned examples
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for example in cleaned_examples:
            writer.writerow(example)
    
    return stats


def clean_all_raw_data(
    require_sat: bool = True,
    min_problem_length: int = 20,
    min_reasoning_length: int = 50,
) -> None:
    """
    Clean all raw data files and combine into a single cleaned dataset.
    """
    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory not found: {RAW_DATA_DIR}")
        return
    
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to clean\n")
    
    all_cleaned = []
    total_stats = {
        "total": 0,
        "valid": 0,
        "invalid_syntax": 0,
        "unsat_or_unknown": 0,
        "too_short": 0,
        "kept": 0,
    }
    
    # Clean each file
    for csv_file in csv_files:
        print(f"Cleaning {csv_file.name}...")
        output_path = CLEANED_DIR / f"cleaned_{csv_file.name}"
        
        stats = clean_csv_file(
            csv_file,
            output_path,
            require_sat=require_sat,
            min_problem_length=min_problem_length,
            min_reasoning_length=min_reasoning_length,
        )
        
        # Accumulate stats
        for key in total_stats:
            total_stats[key] += stats[key]
        
        print(f"  Total: {stats['total']}")
        print(f"  Kept: {stats['kept']} ({stats['kept']/max(stats['total'], 1)*100:.1f}%)")
        print(f"  Invalid syntax: {stats['invalid_syntax']}")
        print(f"  UNSAT/Unknown: {stats['unsat_or_unknown']}")
        print(f"  Too short: {stats['too_short']}\n")
        
        # Collect cleaned examples
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if row and len(row) >= 3:
                    all_cleaned.append(row)
    
    # Write combined cleaned dataset
    combined_path = CLEANED_DIR / "cleaned_dataset.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for example in all_cleaned:
            writer.writerow(example)
    
    print(f"\n{'='*60}")
    print(f"Total Statistics:")
    print(f"  Total examples: {total_stats['total']}")
    print(f"  Kept: {total_stats['kept']} ({total_stats['kept']/max(total_stats['total'], 1)*100:.1f}%)")
    print(f"  Invalid syntax: {total_stats['invalid_syntax']}")
    print(f"  UNSAT/Unknown: {total_stats['unsat_or_unknown']}")
    print(f"  Too short: {total_stats['too_short']}")
    print(f"\nCombined cleaned dataset saved to: {combined_path}")


def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean and filter SMT examples")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input CSV file (if not specified, cleans all files in data/raw/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV file",
    )
    parser.add_argument(
        "--allow-unsat",
        action="store_true",
        help="Keep UNSAT examples (default: only SAT)",
    )
    parser.add_argument(
        "--min-problem-length",
        type=int,
        default=20,
        help="Minimum problem statement length (default: 20)",
    )
    parser.add_argument(
        "--min-reasoning-length",
        type=int,
        default=50,
        help="Minimum reasoning length (default: 50)",
    )
    
    args = parser.parse_args()
    
    if args.input:
        # Clean single file
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else CLEANED_DIR / f"cleaned_{input_path.name}"
        
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        
        stats = clean_csv_file(
            input_path,
            output_path,
            require_sat=not args.allow_unsat,
            min_problem_length=args.min_problem_length,
            min_reasoning_length=args.min_reasoning_length,
        )
        
        print(f"\nCleaning complete:")
        print(f"  Total: {stats['total']}")
        print(f"  Kept: {stats['kept']}")
        print(f"  Saved to: {output_path}")
    else:
        # Clean all files
        clean_all_raw_data(
            require_sat=not args.allow_unsat,
            min_problem_length=args.min_problem_length,
            min_reasoning_length=args.min_reasoning_length,
        )


if __name__ == "__main__":
    main()

