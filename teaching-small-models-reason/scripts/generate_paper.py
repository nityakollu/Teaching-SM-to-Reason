"""
Auto-generate paper content from evaluation results.

Reads evaluation results and fills in markdown placeholders.
"""

import json
import re
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper" / "draft"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_evaluation_results() -> Dict:
    """Load evaluation results from JSON files."""
    results = {}
    
    # Try to load comparison file
    comparison_file = EVAL_DIR / "evaluation_comparison.json"
    if comparison_file.exists():
        with comparison_file.open("r", encoding="utf-8") as f:
            results["comparison"] = json.load(f)
    
    # Load individual result files
    for result_file in EVAL_DIR.glob("*_results.json"):
        model_type = result_file.stem.replace("_results", "")
        with result_file.open("r", encoding="utf-8") as f:
            results[model_type] = json.load(f)
    
    return results


def count_examples() -> Dict[str, int]:
    """Count examples in dataset."""
    counts = {
        "total": 0,
        "by_domain": {},
    }
    
    # Count raw examples
    for csv_file in RAW_DIR.glob("*.csv"):
        if csv_file.name == "broken.csv":
            continue
        domain = csv_file.stem
        with csv_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            count = len([l for l in lines if l.strip()])
            counts["by_domain"][domain] = count
            counts["total"] += count
    
    # Count cleaned examples
    cleaned_file = CLEANED_DIR / "cleaned_dataset.csv"
    if cleaned_file.exists():
        with cleaned_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            counts["cleaned"] = len([l for l in lines if l.strip()])
    else:
        counts["cleaned"] = 0
    
    return counts


def extract_metrics(results: Dict) -> Dict:
    """Extract key metrics from evaluation results."""
    metrics = {}
    
    if "comparison" in results:
        comparison = results["comparison"]
        for model_result in comparison:
            model_type = model_result.get("model_type", "unknown")
            metrics[model_type] = {
                "valid_rate": model_result.get("valid_rate", 0) * 100,
                "satisfiable_rate": model_result.get("satisfiable_rate", 0) * 100,
                "avg_latency": model_result.get("avg_latency", 0),
                "avg_iterations": model_result.get("avg_iterations", 0),
            }
    
    return metrics


def fill_placeholders(content: str, values: Dict) -> str:
    """Fill placeholders in markdown content."""
    # Replace [KEY] with values
    for key, value in values.items():
        placeholder = f"[{key}]"
        if placeholder in content:
            content = content.replace(placeholder, str(value))
    
    return content


def generate_intro(results: Dict, counts: Dict) -> str:
    """Generate introduction section."""
    intro_file = PAPER_DIR / "intro.md"
    if not intro_file.exists():
        return ""
    
    content = intro_file.read_text(encoding="utf-8")
    
    # Fill placeholders
    values = {
        "NUMBER": counts.get("cleaned", counts.get("total", 0)),
    }
    
    return fill_placeholders(content, values)


def generate_results(results: Dict, counts: Dict, metrics: Dict) -> str:
    """Generate results section with actual metrics."""
    results_file = PAPER_DIR / "results.md"
    if not results_file.exists():
        return ""
    
    content = results_file.read_text(encoding="utf-8")
    
    # Extract values from metrics
    baseline_metrics = metrics.get("baseline", {})
    distilled_metrics = metrics.get("distilled", {})
    refined_metrics = metrics.get("distilled_refined", {})
    
    # Calculate improvements
    baseline_valid = baseline_metrics.get("valid_rate", 0)
    distilled_valid = distilled_metrics.get("valid_rate", 0)
    refined_valid = refined_metrics.get("valid_rate", 0)
    
    valid_improvement = distilled_valid - baseline_valid if baseline_valid > 0 else 0
    refined_improvement = refined_valid - distilled_valid if distilled_valid > 0 else 0
    
    values = {
        "NUMBER": counts.get("cleaned", 0),
        "X": f"{baseline_valid:.1f}",
        "Y": f"{baseline_metrics.get('satisfiable_rate', 0):.1f}",
        "Z": f"{100 - baseline_valid:.1f}",
        "TIME": f"{baseline_metrics.get('avg_latency', 0):.2f}",
        "X'": f"{distilled_valid:.1f}",
        "Y'": f"{distilled_metrics.get('satisfiable_rate', 0):.1f}",
        "Z'": f"{100 - distilled_valid:.1f}",
        "TIME'": f"{distilled_metrics.get('avg_latency', 0):.2f}",
        "DELTA": f"{valid_improvement:.1f}",
        "X''": f"{refined_valid:.1f}",
        "Y''": f"{refined_metrics.get('satisfiable_rate', 0):.1f}",
        "TIME''": f"{refined_metrics.get('avg_latency', 0):.2f}",
        "N": f"{refined_metrics.get('avg_iterations', 0):.1f}",
        "X1": f"{distilled_valid:.1f}",
        "Y1": f"{distilled_metrics.get('satisfiable_rate', 0):.1f}",
        "T2": f"{distilled_metrics.get('avg_latency', 0):.2f}",
        "X2": f"{refined_valid:.1f}",
        "Y2": f"{refined_metrics.get('satisfiable_rate', 0):.1f}",
        "T3": f"{refined_metrics.get('avg_latency', 0):.2f}",
        "I": f"{refined_metrics.get('avg_iterations', 0):.1f}",
        "T1": f"{baseline_metrics.get('avg_latency', 0):.2f}",
    }
    
    return fill_placeholders(content, values)


def main():
    """Main function to generate paper content."""
    print("="*60)
    print("Paper Auto-Generation")
    print("="*60)
    
    # Load results
    print("Loading evaluation results...")
    results = load_evaluation_results()
    metrics = extract_metrics(results)
    counts = count_examples()
    
    print(f"Found {len(results)} result file(s)")
    print(f"Dataset: {counts.get('cleaned', counts.get('total', 0))} examples")
    print()
    
    # Generate filled sections
    print("Generating filled sections...")
    
    # Create output directory
    output_dir = PAPER_DIR / "generated"
    output_dir.mkdir(exist_ok=True)
    
    # Generate intro
    intro_content = generate_intro(results, counts)
    if intro_content:
        (output_dir / "intro_filled.md").write_text(intro_content, encoding="utf-8")
        print("  ✓ Generated intro_filled.md")
    
    # Generate results
    results_content = generate_results(results, counts, metrics)
    if results_content:
        (output_dir / "results_filled.md").write_text(results_content, encoding="utf-8")
        print("  ✓ Generated results_filled.md")
    
    # Copy methods and conclusion (no placeholders to fill)
    for section in ["methods.md", "conclusion.md"]:
        src = PAPER_DIR / section
        if src.exists():
            import shutil
            shutil.copy(src, output_dir / section)
            print(f"  ✓ Copied {section}")
    
    print(f"\n✓ Generated paper content in {output_dir}")
    print("\nNote: Some placeholders may remain if evaluation results are not available.")
    print("Run evaluation scripts first to populate metrics.")


if __name__ == "__main__":
    main()

