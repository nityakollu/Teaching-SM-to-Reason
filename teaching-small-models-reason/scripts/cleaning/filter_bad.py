"""
Filter bad SMT-LIB outputs from raw domain CSVs.

Reads all domain CSVs in data/raw/ and detects:
- Valid SMT programs (proper syntax, balanced parentheses, required components)
- Broken programs (missing parentheses, syntax errors, empty)

Writes:
- data/cleaned/valid.csv: All valid examples
- data/raw/broken.csv: All broken examples
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
VALID_OUTPUT = CLEANED_DATA_DIR / "valid.csv"
BROKEN_OUTPUT = RAW_DATA_DIR / "broken.csv"


def check_balanced_parentheses(program: str) -> bool:
    """
    Check if parentheses are balanced in the SMT program.
    
    Returns:
        True if parentheses are balanced, False otherwise
    """
    stack = []
    for char in program:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0


def has_valid_smt_structure(program: str) -> bool:
    """
    Check if program has basic valid SMT-LIB structure.
    
    Returns:
        True if has valid structure, False otherwise
    """
    # Check for set-logic (optional but common)
    # Check for declare-const or declare-fun
    has_declare = bool(re.search(r'\(declare-(const|fun)', program, re.IGNORECASE))
    
    # Check for assert
    has_assert = bool(re.search(r'\(assert', program, re.IGNORECASE))
    
    # Check for check-sat
    has_check_sat = bool(re.search(r'\(check-sat\)', program, re.IGNORECASE))
    
    # Valid SMT program should have at least declare and assert
    # check-sat is highly recommended but not strictly required
    return has_declare and has_assert


def detect_syntax_errors(program: str) -> List[str]:
    """
    Detect common syntax errors in SMT-LIB programs.
    
    Returns:
        List of detected error messages
    """
    errors = []
    
    # Check for balanced parentheses
    if not check_balanced_parentheses(program):
        errors.append("Unbalanced parentheses")
    
    # Check for common invalid patterns
    # Multiple set-logic declarations (should be at most one)
    set_logic_count = len(re.findall(r'\(set-logic', program, re.IGNORECASE))
    if set_logic_count > 1:
        errors.append(f"Multiple set-logic declarations ({set_logic_count})")
    
    # Check for invalid declare patterns (missing closing parens)
    declare_patterns = re.findall(r'\(declare-(const|fun)[^)]*', program, re.IGNORECASE)
    for pattern in declare_patterns:
        if not re.search(r'\)', program[program.find(pattern):program.find(pattern)+200]):
            errors.append("Incomplete declare statement")
    
    # Check for invalid assert patterns
    assert_patterns = re.findall(r'\(assert[^)]*', program, re.IGNORECASE)
    for pattern in assert_patterns:
        # Check if assert has proper closing
        if not re.search(r'\)', program[program.find(pattern):program.find(pattern)+500]):
            errors.append("Incomplete assert statement")
    
    return errors


def is_valid_smt_program(program: str) -> Tuple[bool, str]:
    """
    Determine if an SMT program is valid.
    
    Returns:
        Tuple of (is_valid, reason)
    """
    if not program or not program.strip():
        return False, "Empty program"
    
    program = program.strip()
    
    # Check minimum length
    if len(program) < 20:
        return False, "Program too short"
    
    # Check for valid SMT structure
    if not has_valid_smt_structure(program):
        return False, "Missing required SMT-LIB components (declare-const/fun and assert)"
    
    # Check for syntax errors
    syntax_errors = detect_syntax_errors(program)
    if syntax_errors:
        return False, "; ".join(syntax_errors)
    
    # Additional validation: check for reasonable structure
    # Should have some content between parentheses
    if program.count('(') == 0:
        return False, "No SMT-LIB structure detected"
    
    # Check for reasonable balance (allow small mismatches due to string literals)
    open_parens = program.count('(')
    close_parens = program.count(')')
    if abs(open_parens - close_parens) > 10:  # Allow some mismatch for string literals
        return False, f"Severely unbalanced parentheses ({open_parens} open, {close_parens} close)"
    
    return True, "Valid"


def read_csv_file(csv_path: Path) -> List[Tuple[str, str, str]]:
    """
    Read examples from a CSV file.
    
    Returns:
        List of (problem, reasoning, program) tuples
    """
    examples = []
    
    if not csv_path.exists():
        return examples
    
    with csv_path.open("r", encoding="utf-8") as f:
        # Detect delimiter
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            # Handle different possible column names
            problem = row.get('problem', row.get('Problem', '')).strip()
            reasoning = row.get('reasoning', row.get('Reasoning', '')).strip()
            program = row.get('program', row.get('Program', '')).strip()
            
            # Skip header row if it appears in data
            if problem.lower() == 'problem' and reasoning.lower() == 'reasoning':
                continue
            
            if problem or reasoning or program:
                examples.append((problem, reasoning, program))
    
    return examples


def filter_all_raw_data() -> Dict[str, int]:
    """
    Filter all raw CSV files and separate valid from broken examples.
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total': 0,
        'valid': 0,
        'broken': 0,
    }
    
    valid_examples = []
    broken_examples = []
    
    # Find all CSV files in raw directory (excluding broken.csv)
    csv_files = [f for f in RAW_DATA_DIR.glob("*.csv") if f.name != "broken.csv"]
    
    if not csv_files:
        print(f"No CSV files found in {RAW_DATA_DIR}")
        return stats
    
    print(f"Found {len(csv_files)} CSV file(s) to filter\n")
    
    # Process each file
    for csv_path in csv_files:
        print(f"Processing {csv_path.name}...")
        examples = read_csv_file(csv_path)
        
        file_valid = 0
        file_broken = 0
        
        for problem, reasoning, program in examples:
            stats['total'] += 1
            is_valid, reason = is_valid_smt_program(program)
            
            if is_valid:
                stats['valid'] += 1
                file_valid += 1
                valid_examples.append((problem, reasoning, program))
            else:
                stats['broken'] += 1
                file_broken += 1
                broken_examples.append((problem, reasoning, program, reason))
        
        print(f"  Total: {len(examples)}")
        print(f"  Valid: {file_valid}")
        print(f"  Broken: {file_broken}")
    
    # Write valid examples to data/cleaned/valid.csv
    if valid_examples:
        CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with VALID_OUTPUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["problem", "reasoning", "program"])
            
            for problem, reasoning, program in valid_examples:
                writer.writerow([problem, reasoning, program])
        
        print(f"\n✓ Saved {len(valid_examples)} valid examples to {VALID_OUTPUT}")
    
    # Write broken examples to data/raw/broken.csv
    if broken_examples:
        BROKEN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        
        with BROKEN_OUTPUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["problem", "reasoning", "program", "error"])
            
            for problem, reasoning, program, error in broken_examples:
                writer.writerow([problem, reasoning, program, error])
        
        print(f"✓ Saved {len(broken_examples)} broken examples to {BROKEN_OUTPUT}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Filtering Summary")
    print(f"{'='*60}")
    print(f"Total examples: {stats['total']}")
    print(f"Valid: {stats['valid']} ({stats['valid']/max(stats['total'],1)*100:.1f}%)")
    print(f"Broken: {stats['broken']} ({stats['broken']/max(stats['total'],1)*100:.1f}%)")
    
    return stats


def main():
    """Main function."""
    print("="*60)
    print("SMT-LIB Program Filtering")
    print("="*60)
    print()
    
    stats = filter_all_raw_data()
    
    print(f"\n✓ Filtering complete!")
    print(f"\nOutput files:")
    print(f"  Valid examples: {VALID_OUTPUT}")
    print(f"  Broken examples: {BROKEN_OUTPUT}")


if __name__ == "__main__":
    main()
