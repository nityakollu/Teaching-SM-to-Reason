"""
Analyze scheduling.csv content and identify bad examples.
Works with content visible in editor (even if file not saved).
"""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


# Content from read_file - this is the actual file content
CONTENT = """Assign 3 classes (C1,C2,C3) to 3 time slots (1=09:00–10:00, 2=10:00–11:00, 3=11:00–12:00) in one room. C1 and C3 share an instructor, so they cannot overlap; the room hosts at most one class per slot; C2 cannot be at slot 2.	This problem describes an individual-based relationship constraint where direct connections between certain pairs and exclusion from others are to be enforced. Conditional constraints exist as well, such that if one person is friends with another through mutual acquaintances, it should reflect their connection directly too. SMT solvers can express these relationships using logical operators based on the information given.	"(declare-const ex1_C1 Int)
(declare-const ex1_C2 Int)
(declare-const ex1_C3 Int)
(assert (and (<= 1 ex1_C1) (<= ex1_C1 3)))
(assert (and (<= 1 ex1_C2) (<= ex1_C2 3)))
(assert (and (<= 1 ex1_C3) (<= ex1_C3 3)))
(assert (distinct ex1_C1 ex1_C2 ex1_C3))
(assert (not (= ex1_C2 2)))
(assert (not (= ex1_C1 ex1_C3)))
(check-sat)"
Staff A,B,C cover two shifts per day (S1,S2). Each worker does at most one shift; each shift needs exactly two workers.	Use Booleans A1,B1,C1 for shift S1 and A2,B2,C2 for S2. At-most-one per worker across the two shifts; exactly-two per shift encoded via disjunctions over combinations.	"(declare-const ex2_A1 Bool)
(declare-const ex2_B1 Bool)
(declare-const ex2_C1 Bool)
(declare-const ex2_A2 Bool)
(declare-const ex2_B2 Bool)
(declare-const ex2_C2 Bool)
; at-most-one shift per worker
(assert (not (and ex2_A1 ex2_A2)))
(assert (not (and ex2_B1 ex2_B2)))
(assert (not (and ex2_C1 ex2_C2)))
; exactly two workers on S1
(assert (or (and ex2_A1 ex2_B1 (not ex2_C1))
(and ex2_A1 ex2_C1 (not ex2_B1))
(and ex2_B1 ex2_C1 (not ex2_A1))))
; exactly two workers on S2
(assert (or (and ex2_A2 ex2_B2 (not ex2_C2))
(and ex2_A2 ex2_C2 (not ex2_B2))
(and ex2_B2 ex2_C2 (not ex2_A2))))
(check-sat)"
Two courses (A,B) choose rooms R1 or R2. Types: type(R1)=1 (lecture), type(R2)=2 (lab). A requires lab (type 2); B can use either. If they share a time slot, they must not share the same room.	(declare-const ex50_slot Int) (assert (and (<= 5 ex50_slot) (<= ex50_slot 8))) (check-sat)	"(declare-const ex7_rA Int)
(declare-const ex7_rB Int)
(declare-const ex7_tA Int)
(declare-const ex7_tB Int)
(define-fun ex7_room_type ((r Int)) Int (ite (= r 1) 1 2))
(assert (and (<= 1 ex7_rA) (<= ex7_rA 2)))
(assert (and (<= 1 ex7_rB) (<= ex7_rB 2)))
(assert (and (<= 1 ex7_tA) (<= ex7_tA 2)))
(assert (and (<= 1 ex7_tB) (<= ex7_tB 2)))
; A requires lab type 2
(assert (= (ex7_room_type ex7_rA) 2))
; If same time then different rooms
(assert (=> (= ex7_tA ex7_tB) (not (= ex7_rA ex7_rB))))
(check-sat)"
"""


def main():
    """Analyze content and find bad examples."""
    print("Analyzing scheduling.csv content...")
    print("=" * 80)
    
    # Read from actual file if it exists and has content
    file_path = PROJECT_ROOT / "data" / "raw" / "scheduling.csv"
    content = ""
    
    if file_path.exists() and file_path.stat().st_size > 0:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
        print(f"Read {len(content)} bytes from file")
    else:
        print("File is empty or not saved. Using content from editor view...")
        # For now, we'll work with a sample - user should save file first
        print("Please save scheduling.csv first, then run check_scheduling_simple.py")
        return
    
    if not content.strip():
        print("No content found.")
        return
    
    # Split by newlines and process
    lines = content.split('\n')
    bad_examples = []
    delimiter = "\t"
    
    row_num = 0
    for line in lines:
        if not line.strip():
            continue
        
        row_num += 1
        parts = line.split(delimiter)
        
        if len(parts) < 3:
            # Try comma
            if "," in line and "\t" not in line:
                parts = line.split(",")
                delimiter = ","
            else:
                continue
        
        if len(parts) < 3:
            continue
        
        problem = parts[0].strip()
        reasoning = parts[1].strip()
        program = "".join(parts[2:]).strip()
        
        # Remove quotes
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
                "program_preview": program[:60] + "..." if len(program) > 60 else program,
                "issues": issues,
                "full_line": line,
            })
    
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
            print(f"  Program preview: {ex['program_preview']}")
            print(f"  Issues: {', '.join(ex['issues'])}")
        
        # Save to incomplete directory
        INCOMPLETE_DIR.mkdir(parents=True, exist_ok=True)
        bad_file = INCOMPLETE_DIR / "scheduling_bad.csv"
        
        with bad_file.open("w", newline="", encoding="utf-8") as wf:
            for ex in bad_examples:
                wf.write(ex['full_line'] + '\n')
        
        print(f"\n✓ Bad examples saved to: {bad_file}")
        print(f"  ({len(bad_examples)} examples)")
        print("\nRun: python scripts/cleaning/repair_smt.py to fix them")
    else:
        print("\n✓ All examples look good!")


if __name__ == "__main__":
    main()

