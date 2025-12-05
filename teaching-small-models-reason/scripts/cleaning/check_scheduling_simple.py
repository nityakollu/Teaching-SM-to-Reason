"""
Simple script to check scheduling.csv for bad examples.
Run this after saving the file in your editor.
"""

import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULING_FILE = PROJECT_ROOT / "data" / "raw" / "scheduling.csv"
INCOMPLETE_DIR = PROJECT_ROOT / "data" / "raw" / "incomplete"


def check_smtlib_in_reasoning(reasoning: str) -> bool:
    """Check if reasoning contains SMT-LIB code (bad sign)."""
    smtlib_keywords = ["declare-const", "assert", "check-sat", "(declare", "(assert"]
    reasoning_lower = reasoning.lower()
    return any(keyword in reasoning_lower for keyword in smtlib_keywords)


def is_valid_smtlib(program: str) -> tuple:
    """Basic validation."""
    if not program or not program.strip():
        return False, "Empty"
    
    program = program.strip().strip('"')
    
    if "declare-const" not in program.lower():
        return False, "No declare-const"
    
    if "assert" not in program.lower():
        return False, "No assert"
    
    if "check-sat" not in program.lower():
        return False, "No check-sat"
    
    # Check parentheses balance
    if program.count("(") != program.count(")"):
        return False, "Unbalanced parentheses"
    
    return True, "OK"


def main():
    """Check scheduling.csv for bad examples."""
    if not SCHEDULING_FILE.exists():
        print(f"File not found: {SCHEDULING_FILE}")
        print("Please save the file in your editor first.")
        return
    
    file_size = SCHEDULING_FILE.stat().st_size
    if file_size == 0:
        print(f"File is empty (0 bytes). Please save the file in your editor first.")
        return
    
    print(f"Checking {SCHEDULING_FILE} ({file_size} bytes)...")
    print("=" * 80)
    
    bad_examples = []
    
    try:
        with SCHEDULING_FILE.open("r", encoding="utf-8") as f:
            # Read entire content
            content = f.read()
            
        if not content.strip():
            print("File appears empty after reading.")
            return
        
        # Split by newlines
        lines = content.split('\n')
        print(f"Found {len(lines)} lines")
        
        # Try tab delimiter
        delimiter = "\t"
        row_num = 0
        
        for line in lines:
            if not line.strip():
                continue
            
            row_num += 1
            parts = line.split(delimiter)
            
            if len(parts) < 3:
                # Try comma
                parts = line.split(",")
                if len(parts) >= 3:
                    delimiter = ","
                else:
                    print(f"Row {row_num}: Cannot parse (only {len(parts)} parts)")
                    continue
            
            problem = parts[0].strip() if len(parts) > 0 else ""
            reasoning = parts[1].strip() if len(parts) > 1 else ""
            program = "".join(parts[2:]).strip() if len(parts) > 2 else ""
            
            # Remove quotes from program
            if program.startswith('"') and program.endswith('"'):
                program = program[1:-1]
            
            issues = []
            
            # Check 1: Empty fields
            if not problem:
                issues.append("Empty problem")
            if not reasoning:
                issues.append("Empty reasoning")
            if not program:
                issues.append("Empty program")
            
            # Check 2: SMT-LIB in reasoning (wrong column)
            if reasoning and check_smtlib_in_reasoning(reasoning):
                issues.append("SMT-LIB code in reasoning column")
            
            # Check 3: Valid SMT-LIB program
            if program:
                is_valid, error = is_valid_smtlib(program)
                if not is_valid:
                    issues.append(f"Invalid SMT-LIB: {error}")
            
            # Check 4: Program too short
            if program and len(program) < 30:
                issues.append("Program suspiciously short")
            
            if issues:
                bad_examples.append({
                    "row": row_num,
                    "problem": problem[:80] + "..." if len(problem) > 80 else problem,
                    "reasoning": reasoning[:80] + "..." if len(reasoning) > 80 else reasoning,
                    "issues": issues,
                })
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Report
    print(f"\nTotal rows checked: {row_num}")
    print(f"Bad examples found: {len(bad_examples)}")
    
    if bad_examples:
        print("\n" + "=" * 80)
        print("BAD EXAMPLES:")
        print("=" * 80)
        
        for ex in bad_examples:
            print(f"\nRow {ex['row']}:")
            print(f"  Problem: {ex['problem']}")
            print(f"  Reasoning: {ex['reasoning']}")
            print(f"  Issues: {', '.join(ex['issues'])}")
        
        # Save to incomplete directory
        INCOMPLETE_DIR.mkdir(parents=True, exist_ok=True)
        bad_file = INCOMPLETE_DIR / "scheduling_bad.csv"
        
        # Re-read and save bad rows
        with SCHEDULING_FILE.open("r", encoding="utf-8") as rf:
            content = rf.read()
            lines = content.split('\n')
            
            with bad_file.open("w", newline="", encoding="utf-8") as wf:
                writer = csv.writer(wf, delimiter=delimiter)
                for ex in bad_examples:
                    row_idx = ex["row"] - 1
                    if row_idx < len(lines) and lines[row_idx].strip():
                        parts = lines[row_idx].split(delimiter)
                        if len(parts) >= 3:
                            writer.writerow(parts)
        
        print(f"\n✓ Bad examples saved to: {bad_file}")
        print("Run: python scripts/cleaning/repair_smt.py to fix them")
    else:
        print("\n✓ All examples look good!")


if __name__ == "__main__":
    main()

