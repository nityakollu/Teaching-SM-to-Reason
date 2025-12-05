"""
Z3-based validator for SMT-LIB programs.

Validates SMT-LIB2 programs by:
1. Loading into Z3 solver
2. Validating syntax & satisfiability
3. Saving results to CSV

Usage:
    python scripts/verify/check_validity.py data/cleaned/repaired.csv --output data/eval/validation_results.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    from z3 import (
        Solver,
        parse_smt2_string,
        sat,
        unsat,
        unknown,
    )
except ImportError:
    print("Error: z3-solver not installed. Install with: pip install z3-solver")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SMTValidator:
    """Validates SMT-LIB2 programs using Z3."""
    
    def validate_program(self, smt_program: str) -> Tuple[bool, Optional[str]]:
        """
        Validate an SMT-LIB2 program using Z3.
        
        Args:
            smt_program: The SMT-LIB2 program as a string
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if program is syntactically valid and can be checked
            - error_message: Error message if invalid, None if valid
        """
        # Clean the program
        smt_program = smt_program.strip()
        
        # Basic checks
        if not smt_program:
            return False, "Empty SMT program"
        
        # Try to parse and check with Z3
        try:
            solver = Solver()
            
            # Parse SMT-LIB2 string
            assertions = parse_smt2_string(smt_program)
            
            # Add all assertions to the solver
            for assertion in assertions:
                solver.add(assertion)
            
            # Check satisfiability
            result = solver.check()
            
            # If we get here, the program is syntactically valid
            # Result can be sat, unsat, or unknown - all are valid outcomes
            return True, None
            
        except Exception as e:
            # Extract error message
            error_msg = str(e)
            # Clean up error message
            if "parse error" in error_msg.lower():
                return False, f"Parse error: {error_msg}"
            elif "unknown function" in error_msg.lower():
                return False, f"Unknown function: {error_msg}"
            elif "type error" in error_msg.lower():
                return False, f"Type error: {error_msg}"
            else:
                return False, f"Validation error: {error_msg}"


def validate_csv_file(input_file: Path, output_file: Path) -> None:
    """
    Validate all SMT programs in a CSV file and save results.
    
    Args:
        input_file: Path to input CSV file (format: problem,reasoning,program)
        output_file: Path to output CSV file (format: id,valid,error)
    """
    validator = SMTValidator()
    results = []
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Reading from: {input_file}")
    
    # Read input CSV
    with input_file.open("r", encoding="utf-8") as f:
        # Detect delimiter
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Handle different possible column names
        for row_id, row in enumerate(reader, start=1):
            # Get program from various possible column names
            program = (
                row.get('program', '') or 
                row.get('Program', '') or 
                row.get('smt_program', '') or
                row.get('SMT_PROGRAM', '')
            ).strip()
            
            # If no program column found, try positional (third column)
            if not program and len(row) >= 3:
                program = list(row.values())[2].strip()
            
            if not program:
                results.append({
                    'id': row_id,
                    'valid': False,
                    'error': 'Missing program column'
                })
                continue
            
            # Validate program
            is_valid, error_msg = validator.validate_program(program)
            
            results.append({
                'id': row_id,
                'valid': is_valid,
                'error': error_msg if error_msg else ''
            })
            
            # Progress indicator
            if row_id % 100 == 0:
                print(f"  Processed {row_id} programs...")
    
    # Write results to output CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "valid", "error"])
        
        for result in results:
            writer.writerow([
                result['id'],
                result['valid'],
                result['error']
            ])
    
    # Print summary
    total = len(results)
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = total - valid_count
    
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"Total programs: {total}")
    print(f"Valid: {valid_count} ({valid_count/max(total,1)*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/max(total,1)*100:.1f}%)")
    print(f"\nResults saved to: {output_file}")


def main():
    """Command-line interface for validation."""
    parser = argparse.ArgumentParser(
        description="Validate SMT-LIB2 programs using Z3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify/check_validity.py data/cleaned/repaired.csv --output data/eval/validation_results.csv
        """
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input CSV file containing SMT programs (format: problem,reasoning,program)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output CSV file for validation results (format: id,valid,error)",
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Make output path relative to project root if needed
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    
    # Validate
    validate_csv_file(input_path, output_path)


if __name__ == "__main__":
    main()
