#!/usr/bin/env python3
"""
Complete integrated pipeline runner for the Teaching Small Models to Reason project.

Runs all steps in sequence:
1. Dataset generation
2. Filtering
3. Template repair
4. Z3 validation
5. Format JSONL
6. LoRA training
7. Iterative refinements
8. Evaluation
9. Generate paper draft

All within one integrated environment.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_command(cmd: list, description: str, continue_on_error: bool = False) -> bool:
    """
    Run a command and handle errors.
    
    Args:
        cmd: Command to run as list
        description: Description of what's being done
        continue_on_error: Whether to continue on error
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Running: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=False,  # Show output in real-time
        )
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed with exit code {e.returncode} ({elapsed:.1f}s)")
        
        if continue_on_error:
            response = input("\nContinue despite failure? (y/n): ").lower().strip()
            return response == 'y'
        else:
            print("Stopping pipeline due to error.")
            return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return False


def check_prerequisites() -> bool:
    """Check if prerequisites are met."""
    print("Checking prerequisites...")
    
    # Check if Ollama is available
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("✓ Ollama is available")
        else:
            print("✗ Ollama is not available")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Ollama is not installed or not running")
        print("  Install from: https://ollama.ai/")
        return False
    
    # Check Python dependencies
    try:
        import z3
        import transformers
        import peft
        print("✓ Python dependencies are available")
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e}")
        print("  Install with: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run the complete Teaching Small Models to Reason pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/run_pipeline.py
  
  # Skip generation (use existing data)
  python scripts/run_pipeline.py --skip-generation
  
  # Skip training (only generate data and evaluate baseline)
  python scripts/run_pipeline.py --skip-training
  
  # Use Llama 3 instead of Phi-3
  python scripts/run_pipeline.py --model llama3
        """
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip dataset generation step",
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help="Skip filtering step",
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="Skip template repair step",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip Z3 validation step",
    )
    parser.add_argument(
        "--skip-formatting",
        action="store_true",
        help="Skip JSONL formatting step",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model fine-tuning step",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip iterative refinement step",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step",
    )
    parser.add_argument(
        "--skip-paper",
        action="store_true",
        help="Skip paper generation step",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi3",
        help="Model to use (phi3, llama3, or HuggingFace path) (default: phi3)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--no-prereq-check",
        action="store_true",
        help="Skip prerequisite checking",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if a step fails (prompts for confirmation)",
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("TEACHING SMALL MODELS TO REASON - COMPLETE PIPELINE")
    print("="*70)
    print(f"Starting pipeline execution...")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Check prerequisites
    if not args.no_prereq_check:
        if not check_prerequisites():
            print("\n✗ Prerequisites check failed. Please fix issues and try again.")
            sys.exit(1)
        print()
    
    steps_completed = []
    steps_failed = []
    continue_on_error = args.continue_on_error
    
    # Step 1: Dataset Generation
    if not args.skip_generation:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "dataset_generation" / "generate.py"),
        ]
        if run_command(cmd, "1. Dataset Generation", continue_on_error):
            steps_completed.append("Dataset Generation")
        else:
            steps_failed.append("Dataset Generation")
            if not continue_on_error:
                sys.exit(1)
    else:
        print("⏭ Skipping dataset generation")
    
    # Step 2: Filtering
    if not args.skip_filtering:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "cleaning" / "filter_bad.py"),
        ]
        if run_command(cmd, "2. Filtering Bad Outputs", continue_on_error):
            steps_completed.append("Filtering")
        else:
            steps_failed.append("Filtering")
            if not continue_on_error:
                sys.exit(1)
    else:
        print("⏭ Skipping filtering")
    
    # Step 3: Template Repair
    if not args.skip_repair:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "cleaning" / "repair_smt.py"),
        ]
        if run_command(cmd, "3. Template-Based SMT Repair", continue_on_error):
            steps_completed.append("Template Repair")
        else:
            steps_failed.append("Template Repair")
            if not continue_on_error:
                sys.exit(1)
    else:
        print("⏭ Skipping template repair")
    
    # Step 4: Z3 Validation
    if not args.skip_validation:
        cleaned_dataset = PROJECT_ROOT / "data" / "cleaned" / "cleaned_dataset.csv"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "verify" / "check_validity.py"),
            str(cleaned_dataset),
        ]
        if run_command(cmd, "4. Z3 Validation", continue_on_error):
            steps_completed.append("Z3 Validation")
        else:
            steps_failed.append("Z3 Validation")
            if not continue_on_error:
                sys.exit(1)
    else:
        print("⏭ Skipping Z3 validation")
    
    # Step 5: Format JSONL
    if not args.skip_formatting:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "distillation" / "format_jsonl.py"),
        ]
        if run_command(cmd, "5. Format JSONL for Training", continue_on_error):
            steps_completed.append("JSONL Formatting")
        else:
            steps_failed.append("JSONL Formatting")
            if not continue_on_error:
                sys.exit(1)
    else:
        print("⏭ Skipping JSONL formatting")
    
    # Step 6: LoRA Training
    if not args.skip_training:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "distillation" / "finetune.py"),
            "--model", args.model,
            "--epochs", str(args.epochs),
        ]
        if run_command(cmd, f"6. LoRA Fine-tuning ({args.model})", continue_on_error):
            steps_completed.append("LoRA Training")
        else:
            steps_failed.append("LoRA Training")
            print("⚠ Training failed - evaluation may use baseline model")
    else:
        print("⏭ Skipping LoRA training")
    
    # Step 7: Iterative Refinement
    if not args.skip_refinement:
        training_jsonl = PROJECT_ROOT / "data" / "cleaned" / "training.jsonl"
        if training_jsonl.exists():
            refined_output = PROJECT_ROOT / "data" / "cleaned" / "refined.jsonl"
            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / "refine" / "refiner.py"),
                str(training_jsonl),
                "--output", str(refined_output),
            ]
            if run_command(cmd, "7. Iterative Refinement", continue_on_error):
                steps_completed.append("Iterative Refinement")
            else:
                steps_failed.append("Iterative Refinement")
        else:
            print("⚠ Skipping refinement - training.jsonl not found")
    else:
        print("⏭ Skipping iterative refinement")
    
    # Step 8: Evaluation
    if not args.skip_evaluation:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "eval" / "evaluate.py"),
            "--all",  # Evaluate baseline, distilled, and refined
        ]
        
        # Add model path if available
        distilled_model = PROJECT_ROOT / "models" / "distilled" / args.model.replace("/", "_") / "final"
        if distilled_model.exists():
            cmd.extend(["--model", str(distilled_model)])
        
        if run_command(cmd, "8. Evaluation", continue_on_error):
            steps_completed.append("Evaluation")
        else:
            steps_failed.append("Evaluation")
    else:
        print("⏭ Skipping evaluation")
    
    # Step 9: Generate Paper Draft
    if not args.skip_paper:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "generate_paper.py"),
        ]
        if run_command(cmd, "9. Generate Paper Draft", continue_on_error):
            steps_completed.append("Paper Generation")
        else:
            steps_failed.append("Paper Generation")
    else:
        print("⏭ Skipping paper generation")
    
    # Final Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"\nCompleted steps ({len(steps_completed)}/{9}):")
    for i, step in enumerate(steps_completed, 1):
        print(f"  {i}. ✓ {step}")
    
    if steps_failed:
        print(f"\nFailed steps ({len(steps_failed)}):")
        for step in steps_failed:
            print(f"  ✗ {step}")
    
    print(f"\n{'='*70}")
    if steps_failed:
        print("⚠ Pipeline completed with some failures. Check logs above.")
    else:
        print("✓ Pipeline completed successfully!")
    
    print(f"\nResults:")
    print(f"  - Dataset: {PROJECT_ROOT / 'data' / 'cleaned'}")
    print(f"  - Models: {PROJECT_ROOT / 'models' / 'distilled'}")
    print(f"  - Evaluation: {PROJECT_ROOT / 'data' / 'eval'}")
    print(f"  - Paper: {PROJECT_ROOT / 'paper' / 'draft' / 'generated'}")
    print()


if __name__ == "__main__":
    main()
