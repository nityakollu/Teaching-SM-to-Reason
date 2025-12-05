import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests  # make sure requests is installed: pip install requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# CHANGE THIS TO THE MODEL YOU HAVE INSTALLED IN OLLAMA
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = "http://localhost:11434"


DOMAINS: Dict[str, str] = {
    "scheduling": "scheduling.csv",
    "puzzles": "puzzles.csv",
    "resource_allocation": "resource_alloc.csv",
    "relationships": "relationships.csv",
}


@dataclass
class Example:
    problem: str
    reasoning: str
    program: str
    domain: str


# ------------------------------------------------------------
# CHECK OLLAMA CONNECTION
# ------------------------------------------------------------
def check_ollama_available() -> bool:
    """Check that Ollama is running and model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama responded but not OK:", response.text)
            return False

        models = response.json().get("models", [])
        names = [m.get("name", "") for m in models]

        # Match prefix (phi3 matches phi3:instruct)
        for m in names:
            if OLLAMA_MODEL.split(":")[0] in m:
                return True

        print(f"❌ Model '{OLLAMA_MODEL}' not found. Available models:", names)
        return False

    except Exception as e:
        print("❌ Error connecting to Ollama:", e)
        return False


# ------------------------------------------------------------
# PROMPT GENERATOR
# ------------------------------------------------------------
def build_prompt(domain: str) -> str:
    domain_human = {
        "resource_allocation": "resource allocation",
        "scheduling": "scheduling",
        "puzzles": "logic puzzles",
        "relationships": "relationships / social constraints",
    }[domain]

    return f"""
You are generating ONE high-quality constraint reasoning problem for SMT-LIB.

Domain: {domain_human}

Return EXACTLY ONE JSON object with fields:
{{
  "problem": "...",
  "reasoning": "...",
  "program_smtlib": "SMT-LIB code here"
}}

Rules:
- Produce valid JSON ONLY (no markdown, no backticks)
- The SMT-LIB program must be syntactically valid and include:
  - (set-logic QF_UF) or QF_LIA
  - (declare-const ...)
  - (assert ...)
  - (check-sat)
- Use simple integer or boolean constraints.
- Use \\n for newlines inside the SMT string.
""".strip()


# ------------------------------------------------------------
# JSON PARSER
# ------------------------------------------------------------
def parse_example_from_json(raw: str, domain: str) -> Example:
    raw = raw.strip()

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")

    if first_brace == -1 or last_brace == -1:
        raise ValueError("No JSON object detected.")

    candidate = raw[first_brace:last_brace+1]

    data = json.loads(candidate)

    problem = str(data.get("problem", "")).strip()
    reasoning = str(data.get("reasoning", "")).strip()
    program = str(data.get("program_smtlib", "")).strip()

    if not problem or not reasoning or not program:
        raise ValueError("Missing one of: problem, reasoning, program_smtlib")

    return Example(problem=problem, reasoning=reasoning, program=program, domain=domain)


# ------------------------------------------------------------
# CALL MODEL WITH RETRIES
# ------------------------------------------------------------
def call_with_retries(fn: Callable[[], str], retries: int = 3) -> str:
    last_error = None
    for attempt in range(1, retries+1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            print(f"⚠️ Retry {attempt}/{retries} failed due to: {e}")
            time.sleep(1.2 * attempt)
    raise last_error


def gen_with_ollama(domain: str) -> str:
    prompt = build_prompt(domain)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 3000
            }
        },
        timeout=600,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    return response.json().get("response", "").strip()


# ------------------------------------------------------------
# MAIN GEN LOOP
# ------------------------------------------------------------
def generate_examples_for_domain(domain: str, n: int) -> List[Example]:
    if not check_ollama_available():
        raise RuntimeError("❌ Ollama not available. Start Ollama and pull model.")

    results: List[Example] = []

    while len(results) < n:
        try:
            raw = call_with_retries(lambda: gen_with_ollama(domain))
            try:
                ex = parse_example_from_json(raw, domain)
                results.append(ex)
                print(f"[{domain}] ✔ Example {len(results)}/{n}")
            except Exception as parse_err:
                print(f"[{domain}] ❌ Parse error: {parse_err}")
                print("--- Raw output was:\n", raw)
                continue

        except Exception as e:
            print(f"[{domain}] ❌ Model call failed: {e}")
            continue

        time.sleep(0.5)

    return results


# ------------------------------------------------------------
# WRITE RESULTS
# ------------------------------------------------------------
def write_examples_to_csv(examples: List[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["problem", "reasoning", "program"])
        for ex in examples:
            writer.writerow([
                ex.problem.replace("\n", " "),
                ex.reasoning.replace("\n", " "),
                ex.program.replace("\n", "\\n")
            ])


# ------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------
def main():
    print("🚀 Starting dataset generation…")

    for domain, filename in DOMAINS.items():
        out_path = RAW_DATA_DIR / filename
        print(f"\n=== GENERATING DOMAIN: {domain} ===")
        print(f"Output file: {out_path}")

        examples = generate_examples_for_domain(domain, n=125)   # Change to 50 later
        write_examples_to_csv(examples, out_path)

        print(f"✅ Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
