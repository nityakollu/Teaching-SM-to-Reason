"""
Deterministic SMT template-based repair (NO LLM, fully deterministic).

Uses templates for:
- Boolean constraints
- Integer constraints
- Equality
- Inequality
- Friend relationships
- Scheduling constraints

Parses natural-language problems and builds minimal valid SMT-LIB programs.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CLEANED_DATA_DIR = PROJECT_ROOT / "data" / "cleaned"
BROKEN_FILE = RAW_DATA_DIR / "broken.csv"
REPAIRED_OUTPUT = CLEANED_DATA_DIR / "repaired.csv"


class SMTTemplateLibrary:
    """Library of deterministic SMT-LIB2 templates for common constraint patterns."""
    
    @staticmethod
    def template_boolean_constraint(var: str, value: bool = True) -> str:
        """Template for boolean constraints."""
        return f"""(set-logic QF_UF)
(declare-const {var} Bool)
(assert (= {var} {str(value).lower()}))
(check-sat)"""
    
    @staticmethod
    def template_integer_constraint(var: str, lower: int = 0, upper: int = 10) -> str:
        """Template for integer constraints with bounds."""
        return f"""(set-logic QF_LIA)
(declare-const {var} Int)
(assert (and (>= {var} {lower}) (<= {var} {upper})))
(check-sat)"""
    
    @staticmethod
    def template_equality(var: str, value: int) -> str:
        """Template for equality constraints."""
        return f"""(set-logic QF_LIA)
(declare-const {var} Int)
(assert (= {var} {value}))
(check-sat)"""
    
    @staticmethod
    def template_inequality(var1: str, var2: str) -> str:
        """Template for inequality constraints (distinct values)."""
        return f"""(set-logic QF_LIA)
(declare-const {var1} Int)
(declare-const {var2} Int)
(assert (and (>= {var1} 0) (<= {var1} 10)))
(assert (and (>= {var2} 0) (<= {var2} 10)))
(assert (distinct {var1} {var2}))
(check-sat)"""
    
    @staticmethod
    def template_friend_relationship(var1: str, var2: str, are_friends: bool = True) -> str:
        """Template for friend relationship constraints."""
        return f"""(set-logic QF_UF)
(declare-const {var1}_friend_{var2} Bool)
(declare-const {var2}_friend_{var1} Bool)
(assert (= {var1}_friend_{var2} {var2}_friend_{var1}))
(assert (= {var1}_friend_{var2} {str(are_friends).lower()}))
(check-sat)"""
    
    @staticmethod
    def template_mutual_friend(var1: str, var2: str, var3: str) -> str:
        """Template for mutual friend relationships."""
        return f"""(set-logic QF_UF)
(declare-const {var1}_friend_{var2} Bool)
(declare-const {var2}_friend_{var1} Bool)
(declare-const {var1}_friend_{var3} Bool)
(declare-const {var3}_friend_{var1} Bool)
(declare-const {var2}_friend_{var3} Bool)
(declare-const {var3}_friend_{var2} Bool)
(assert (= {var1}_friend_{var2} {var2}_friend_{var1}))
(assert (= {var1}_friend_{var3} {var3}_friend_{var1}))
(assert (= {var2}_friend_{var3} {var3}_friend_{var2}))
(assert (and {var1}_friend_{var2} {var2}_friend_{var3}))
(check-sat)"""
    
    @staticmethod
    def template_scheduling_basic(tasks: List[str], slots: int = 3) -> str:
        """Template for basic scheduling with time slots."""
        program = "(set-logic QF_LIA)\n"
        
        for task in tasks:
            program += f"(declare-const {task}_slot Int)\n"
            program += f"(assert (and (>= {task}_slot 1) (<= {task}_slot {slots})))\n"
        
        if len(tasks) > 1:
            program += f"(assert (distinct {' '.join([f'{t}_slot' for t in tasks])}))\n"
        
        program += "(check-sat)"
        return program
    
    @staticmethod
    def template_scheduling_conflict(tasks: List[str], slots: int = 3) -> str:
        """Template for scheduling with conflicts (cannot be together)."""
        program = "(set-logic QF_LIA)\n"
        
        for task in tasks:
            program += f"(declare-const {task}_slot Int)\n"
            program += f"(assert (and (>= {task}_slot 1) (<= {task}_slot {slots})))\n"
        
        # Ensure conflicting tasks are in different slots
        if len(tasks) >= 2:
            program += f"(assert (distinct {' '.join([f'{t}_slot' for t in tasks[:2]])}))\n"
        
        program += "(check-sat)"
        return program
    
    @staticmethod
    def template_scheduling_precedence(task1: str, task2: str, slots: int = 3) -> str:
        """Template for scheduling with precedence constraints."""
        return f"""(set-logic QF_LIA)
(declare-const {task1}_slot Int)
(declare-const {task2}_slot Int)
(assert (and (>= {task1}_slot 1) (<= {task1}_slot {slots})))
(assert (and (>= {task2}_slot 1) (<= {task2}_slot {slots})))
(assert (< {task1}_slot {task2}_slot))
(check-sat)"""
    
    @staticmethod
    def template_integer_arithmetic(var1: str, var2: str, op: str = "+", result: int = 10) -> str:
        """Template for integer arithmetic constraints."""
        if op == "+":
            constraint = f"(= (+ {var1} {var2}) {result})"
        elif op == "-":
            constraint = f"(= (- {var1} {var2}) {result})"
        elif op == "*":
            constraint = f"(= (* {var1} {var2}) {result})"
        else:
            constraint = f"(= (+ {var1} {var2}) {result})"
        
        return f"""(set-logic QF_LIA)
(declare-const {var1} Int)
(declare-const {var2} Int)
(assert (and (>= {var1} 0) (<= {var1} 10)))
(assert (and (>= {var2} 0) (<= {var2} 10)))
(assert {constraint})
(check-sat)"""
    
    @staticmethod
    def template_minimal() -> str:
        """Minimal valid SMT-LIB template as fallback."""
        return """(set-logic QF_LIA)
(declare-const x Int)
(assert (>= x 0))
(check-sat)"""


class ProblemParser:
    """Parse natural language problems using deterministic regex patterns."""
    
    @staticmethod
    def extract_variables(text: str) -> List[str]:
        """Extract variable names from problem text."""
        variables = set()
        
        # Pattern for common variable naming conventions
        patterns = [
            r'\b([A-Z]\d+)\b',  # C1, C2, A1, P1, etc.
            r'\b([A-Z]{1,2})\b',  # A, B, C, AB, etc.
            r'(?:task|class|job|person|room|machine)\s+([A-Z0-9]+)',  # task T1, class C1
            r'([A-Z][a-z]+)\s+(?:must|cannot|should)',  # TaskA must, ClassB cannot
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            variables.update(matches)
        
        # Filter out common words that aren't variables
        exclude_words = {'THE', 'AND', 'OR', 'NOT', 'FOR', 'ALL', 'EXISTS', 'SET', 'LOGIC'}
        variables = {v for v in variables if v.upper() not in exclude_words}
        
        return sorted(list(variables))[:5]  # Limit to 5 variables
    
    @staticmethod
    def extract_numbers(text: str) -> List[int]:
        """Extract numbers from text."""
        matches = re.findall(r'\b\d+\b', text)
        return [int(m) for m in matches]
    
    @staticmethod
    def detect_constraint_type(text: str) -> str:
        """Detect the type of constraint from problem text."""
        text_lower = text.lower()
        
        # Friend/relationship constraints
        if any(word in text_lower for word in ['friend', 'mutual', 'know', 'relationship']):
            if 'not' in text_lower or 'cannot' in text_lower:
                return 'friend_not'
            elif 'mutual' in text_lower:
                return 'mutual_friend'
            else:
                return 'friend'
        
        # Scheduling constraints
        if any(word in text_lower for word in ['schedule', 'slot', 'time', 'meeting', 'class']):
            if 'cannot' in text_lower or 'conflict' in text_lower or 'together' in text_lower:
                return 'scheduling_conflict'
            elif 'before' in text_lower or 'after' in text_lower or 'precedence' in text_lower:
                return 'scheduling_precedence'
            else:
                return 'scheduling'
        
        # Equality constraints
        if any(word in text_lower for word in ['equal', '=', 'same', 'equals']):
            return 'equality'
        
        # Inequality constraints
        if any(word in text_lower for word in ['different', 'distinct', 'not equal', '!=']):
            return 'inequality'
        
        # Integer constraints
        if any(word in text_lower for word in ['sum', '+', 'plus', 'total', 'add']):
            return 'integer_arithmetic'
        
        # Boolean constraints
        if any(word in text_lower for word in ['true', 'false', 'boolean', 'bool']):
            return 'boolean'
        
        # Default: integer constraint
        return 'integer'
    
    @staticmethod
    def detect_domain(text: str) -> str:
        """Detect problem domain."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['schedule', 'time slot', 'class', 'meeting', 'room', 'task']):
            return 'scheduling'
        elif any(word in text_lower for word in ['friend', 'relationship', 'know', 'mutual', 'person']):
            return 'relationships'
        elif any(word in text_lower for word in ['assign', 'resource', 'staff', 'worker', 'allocation']):
            return 'resource_allocation'
        else:
            return 'puzzles'


class SMTRepairer:
    """Repair broken SMT programs using deterministic templates."""
    
    def __init__(self):
        self.templates = SMTTemplateLibrary()
        self.parser = ProblemParser()
    
    def repair(self, problem: str, reasoning: str, broken_program: str) -> str:
        """
        Repair a broken SMT program using deterministic templates.
        
        Args:
            problem: Natural language problem statement
            reasoning: Original reasoning (may be helpful)
            broken_program: Broken or incomplete program
        
        Returns:
            Repaired SMT-LIB2 program
        """
        # Extract information from problem
        variables = self.parser.extract_variables(problem)
        numbers = self.parser.extract_numbers(problem)
        constraint_type = self.parser.detect_constraint_type(problem)
        domain = self.parser.detect_domain(problem)
        
        # Ensure we have at least one variable
        if not variables:
            variables = ['X', 'Y', 'Z']
        
        # Build program based on constraint type and domain
        if constraint_type == 'friend':
            if len(variables) >= 2:
                return self.templates.template_friend_relationship(variables[0], variables[1], True)
            else:
                return self.templates.template_boolean_constraint(variables[0], True)
        
        elif constraint_type == 'friend_not':
            if len(variables) >= 2:
                return self.templates.template_friend_relationship(variables[0], variables[1], False)
            else:
                return self.templates.template_boolean_constraint(variables[0], False)
        
        elif constraint_type == 'mutual_friend':
            if len(variables) >= 3:
                return self.templates.template_mutual_friend(variables[0], variables[1], variables[2])
            elif len(variables) >= 2:
                return self.templates.template_friend_relationship(variables[0], variables[1], True)
            else:
                return self.templates.template_boolean_constraint(variables[0], True)
        
        elif constraint_type == 'scheduling':
            slots = numbers[0] if numbers else 3
            return self.templates.template_scheduling_basic(variables[:3], slots)
        
        elif constraint_type == 'scheduling_conflict':
            slots = numbers[0] if numbers else 3
            return self.templates.template_scheduling_conflict(variables[:2], slots)
        
        elif constraint_type == 'scheduling_precedence':
            slots = numbers[0] if numbers else 3
            if len(variables) >= 2:
                return self.templates.template_scheduling_precedence(variables[0], variables[1], slots)
            else:
                return self.templates.template_scheduling_basic(variables[:2], slots)
        
        elif constraint_type == 'equality':
            value = numbers[0] if numbers else 5
            return self.templates.template_equality(variables[0], value)
        
        elif constraint_type == 'inequality':
            if len(variables) >= 2:
                return self.templates.template_inequality(variables[0], variables[1])
            else:
                return self.templates.template_integer_constraint(variables[0])
        
        elif constraint_type == 'integer_arithmetic':
            result = numbers[0] if numbers else 10
            if len(variables) >= 2:
                return self.templates.template_integer_arithmetic(variables[0], variables[1], "+", result)
            else:
                return self.templates.template_integer_constraint(variables[0])
        
        elif constraint_type == 'boolean':
            return self.templates.template_boolean_constraint(variables[0], True)
        
        else:  # Default: integer constraint
            lower = numbers[0] if numbers else 0
            upper = numbers[1] if len(numbers) > 1 else (numbers[0] + 10 if numbers else 10)
            return self.templates.template_integer_constraint(variables[0], lower, upper)


def repair_broken_examples() -> Dict[str, int]:
    """
    Repair all broken examples from broken.csv.
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'total': 0,
        'repaired': 0,
        'failed': 0,
    }
    
    if not BROKEN_FILE.exists():
        print(f"Broken file not found: {BROKEN_FILE}")
        print("Run filter_bad.py first to identify broken examples.")
        return stats
    
    repairer = SMTRepairer()
    repaired_examples = []
    
    # Read broken examples
    with BROKEN_FILE.open("r", encoding="utf-8") as f:
        # Detect delimiter
        first_line = f.readline()
        f.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            # Handle different possible column names
            problem = row.get('problem', row.get('Problem', '')).strip()
            reasoning = row.get('reasoning', row.get('Reasoning', '')).strip()
            broken_program = row.get('program', row.get('Program', '')).strip()
            
            # Skip header row if it appears in data
            if problem.lower() == 'problem' and reasoning.lower() == 'reasoning':
                continue
            
            if not problem:
                continue
            
            stats['total'] += 1
            
            try:
                repaired_program = repairer.repair(problem, reasoning, broken_program)
                repaired_examples.append((problem, reasoning, repaired_program))
                stats['repaired'] += 1
            except Exception as e:
                print(f"  Error repairing example: {e}")
                stats['failed'] += 1
    
    # Save repaired examples to data/cleaned/repaired.csv
    if repaired_examples:
        def normalize_program(program: str) -> str:
            """Collapse whitespace so each CSV row stays on a single line."""
            return " ".join(program.strip().split())

        CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        with REPAIRED_OUTPUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(["problem", "reasoning", "program"])
            
            for problem, reasoning, program in repaired_examples:
                writer.writerow([problem, reasoning, normalize_program(program)])
        
        print(f"\n✓ Saved {len(repaired_examples)} repaired examples to {REPAIRED_OUTPUT}")
        
        if len(repaired_examples) != stats["total"]:
            print(f"[warn] Output rows ({len(repaired_examples)}) != input rows ({stats['total']}).")
        else:
            print("✓ Output row count matches input row count.")
    
    return stats


def main():
    """Main function."""
    print("="*60)
    print("Deterministic SMT Template-Based Repair")
    print("="*60)
    print()
    
    stats = repair_broken_examples()
    
    print(f"\n{'='*60}")
    print("Repair Summary")
    print(f"{'='*60}")
    print(f"Total broken examples: {stats['total']}")
    print(f"Repaired: {stats['repaired']} ({stats['repaired']/max(stats['total'],1)*100:.1f}%)")
    print(f"Failed: {stats['failed']}")
    print(f"\n✓ Repair complete!")
    print(f"\nOutput file: {REPAIRED_OUTPUT}")


if __name__ == "__main__":
    main()
