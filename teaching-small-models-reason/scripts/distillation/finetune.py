"""
LoRA finetuning script for local HuggingFace models.

Usage:
    python scripts/distillation/finetune.py \
        --model phi-3-mini-4k-instruct \
        --data data/cleaned/training.jsonl \
        --output models/distilled/phi3-lora

Requirements:
    pip install torch transformers datasets peft accelerate
"""

import argparse
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "microsoft/phi-2"
DEFAULT_DATA = PROJECT_ROOT / "data" / "cleaned" / "training.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "distilled" / "phi2-lora"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA finetuning for local models.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Base HuggingFace model to finetune (default: microsoft/phi-2)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help=f"Training JSONL file (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Directory to save PEFT adapter (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    return parser.parse_args()


def format_example(record: Dict[str, str]) -> str:
    """Format a single example into the prompt + completion format."""
    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    output_text = record.get("output", "")

    return (
        f"<s>[INSTRUCTION] {instruction}\n"
        f"[INPUT] {input_text}\n"
        f"[OUTPUT] {output_text}</s>"
    )


def main():
    args = parse_args()
    model_name = args.model
    data_path = Path(args.data)
    output_path = Path(args.output)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=str(data_path))["train"]

    def preprocess_function(examples):
        prompts = []
        for instruction, input_text, output_text in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        ):
            prompts.append(
                format_example(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "output": output_text,
                    }
                )
            )
        return tokenizer(
            prompts,
            truncation=True,
            max_length=2048,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"✓ LoRA adapter saved to {output_path}")


if __name__ == "__main__":
    main()

