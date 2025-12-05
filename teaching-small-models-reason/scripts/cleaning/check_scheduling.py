"""
Check scheduling.csv for bad or incomplete examples.

Identifies examples with:
- Missing or incomplete SMT-LIB programs
- Invalid SMT-LIB syntax
- Missing (check-sat)
- Other issues
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULING_FILE = PROJECT_ROOT / "data" / "raw" / "scheduling.csv"
INCOMPLETE_DIR = PROJECT_ROOT / "data" / "raw" / "incomplete"


def is_valid_smtlib(program: str) -> Tuple[bool, str]:
    """
    Basic validation of SMT-LIB program.
    Returns (is_valid, error_message)
    """
    if not program or not program.strip():
        return False, "Empty program"
    
    program = program.strip()
    
    # Remove quotes if present
    if program.startswith('"') and program.endswith('"'):
        program = program[1:-1]
    
    # Check for basic SMT-LIB structure
    if "declare-const" not in program.lower():
        return False, "Missing declare-const"
    
    if "assert" not in program.lower():
        return False, "Missing assert statements"
    
    # Check for check-sat
    if "check-sat" not in program.lower():
        return False, "Missing (check-sat)"
    
    # Check for balanced parentheses (basic check)
    open_parens = program.count("(")
    close_parens = program.count(")")
    if open_parens != close_parens:
        return False, f"Unbalanced parentheses: {open_parens} open, {close_parens} close"
    
    # Check for common syntax issues
    if re.search(r'\(declare-const\s+\w+\s+[^)]+\)', program) is None:
        # Try to find at least one valid declare-const
        if not re.search(r'declare-const', program, re.IGNORECASE):
            return False, "No valid declare-const found"
    
    # Check for incomplete statements (ending with incomplete parentheses)
    lines = program.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(';'):
            # Check for incomplete statements
            if stripped.count('(') > stripped.count(')'):
                # Might be multi-line, but flag if suspicious
                pass
    
    return True, "OK"


def check_example(row_num: int, problem: str, reasoning: str, program: str) -> Dict:
    """Check a single example and return issues found."""
    issues = []
    
    # Check if fields are empty
    if not problem or not problem.strip():
        issues.append("Empty problem")
    
    if not reasoning or not reasoning.strip():
        issues.append("Empty reasoning")
    
    if not program or not program.strip():
        issues.append("Empty program")
        return {
            "row": row_num,
            "problem": problem[:100] + "..." if len(problem) > 100 else problem,
            "issues": issues,
            "is_bad": True,
        }
    
    # Validate SMT-LIB
    is_valid, error_msg = is_valid_smtlib(program)
    if not is_valid:
        issues.append(f"Invalid SMT-LIB: {error_msg}")
    
    # Check for suspicious patterns
    if len(program) < 50:
        issues.append("Program too short (likely incomplete)")
    
    # Check for common incomplete patterns
    if program.count("declare-const") == 0:
        issues.append("No declare-const statements")
    
    if program.count("assert") == 0:
        issues.append("No assert statements")
    
    # Check if program ends abruptly
    if not program.rstrip().endswith(")"):
        # Might be OK if ends with check-sat, but flag if suspicious
        if not program.rstrip().endswith("(check-sat)"):
            issues.append("Program may be cut off")
    
    return {
        "row": row_num,
        "problem": problem[:100] + "..." if len(problem) > 100 else problem,
        "issues": issues,
        "is_bad": len(issues) > 0,
    }


def main():
    """Check all examples in scheduling.csv."""
    if not SCHEDULING_FILE.exists():
        print(f"Error: {SCHEDULING_FILE} does not exist")
        return
    
    print(f"Checking {SCHEDULING_FILE}...")
    print("=" * 80)
    
    bad_examples = []
    all_examples = []
    
    # Read the file (tab-separated)
    # Handle both single-line and multi-line formats
    try:
        with SCHEDULING_FILE.open("r", encoding="utf-8") as f:
            content = f.read()
            
        # If file appears empty, try reading with read_file approach
        if not content or len(content.strip()) == 0:
            print("Warning: File appears empty when read directly.")
            print("Trying alternative reading method...")
            # Read line by line
            with SCHEDULING_FILE.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        
        if not content or len(content.strip()) == 0:
            print(f"Error: Could not read content from {SCHEDULING_FILE}")
            return
        
        # Split by newlines first
        lines = content.split('\n')
        if len(lines) == 1:
            # Might be all on one line with tabs - try splitting by pattern
            # Look for tab-separated entries
            if '\t' in content:
                # Try to split by looking for patterns like "Assign" or problem starts
                # For now, treat as single row if it's one line
                lines = [content]
        
        delimiter = "\t" if "\t" in content else ","
        
        row_num = 0
        for line in lines:
            if not line.strip():
                continue
            
            row_num += 1
            # Split by delimiter
            parts = line.split(delimiter)
            
            if len(parts) < 3:
                # Try to handle cases where program spans multiple "lines" in the same cell
                # Look for the pattern: problem, reasoning, then program (which may have newlines)
                # For tab-separated, we expect exactly 3 parts
                if delimiter == "\t" and len(parts) >= 2:
                    # Program might be in remaining parts
                    problem = parts[0].strip()
                    reasoning = parts[1].strip()
                    program = delimiter.join(parts[2:]).strip() if len(parts) > 2 else ""
                else:
                    print(f"Row {row_num}: Insufficient columns ({len(parts)}), skipping")
                    continue
            else:
                problem = parts[0].strip()
                reasoning = parts[1].strip()
                program = delimiter.join(parts[2:]).strip()  # Join remaining parts as program
            
            # Clean up program (remove quotes if present)
            if program.startswith('"') and program.endswith('"'):
                program = program[1:-1]
            
            result = check_example(row_num, problem, reasoning, program)
            all_examples.append(result)
            
            if result["is_bad"]:
                bad_examples.append(result)
                
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Report results
    print(f"\nTotal examples checked: {len(all_examples)}")
    print(f"Bad examples found: {len(bad_examples)}")
    print(f"Good examples: {len(all_examples) - len(bad_examples)}")
    
    if bad_examples:
        print("\n" + "=" * 80)
        print("BAD EXAMPLES FOUND:")
        print("=" * 80)
        
        for ex in bad_examples:
            print(f"\nRow {ex['row']}:")
            print(f"  Problem: {ex['problem']}")
            print(f"  Issues: {', '.join(ex['issues'])}")
        
        # Ask if user wants to move bad examples to incomplete directory
        print("\n" + "=" * 80)
        print(f"Found {len(bad_examples)} bad example(s).")
        print("You can move them to data/raw/incomplete/ for repair.")
        
        # Save bad examples to incomplete directory
        INCOMPLETE_DIR.mkdir(parents=True, exist_ok=True)
        bad_file = INCOMPLETE_DIR / "scheduling_bad.csv"
        
        with bad_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for ex in bad_examples:
                # Get full data for bad examples
                row_idx = ex["row"] - 1
                if row_idx < len(all_examples):
                    # Re-read to get full data
                    with SCHEDULING_FILE.open("r", encoding="utf-8") as rf:
                        reader = csv.reader(rf, delimiter=delimiter)
                        rows = list(reader)
                        if row_idx < len(rows):
                            writer.writerow(rows[row_idx])
        
        print(f"\nBad examples saved to: {bad_file}")
        print("Run repair_smt.py to fix them.")
    else:
        print("\n✓ All examples look good!")


if __name__ == "__main__":
    main()

