import logging
import os
import random
import argparse
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, IA3Config
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################
# Prompt Template
############################################
PROMPT_NO_INPUT = (
    "Below is a grade-school math problem. Please work through the reasoning "
    "step-by-step (chain-of-thought) and then provide the final numeric answer "
    "on a new line preceded by '####'.\n\n"
    "### Instruction:\n{instruction}\n\n"
)

############################################
# Dataset Preprocessing
############################################
def preprocess_gsm8k(example: Dict[str, str]) -> Dict[str, str]:
    prompt_str = PROMPT_NO_INPUT.format(instruction=example["question"])
    return {
        "inputs": prompt_str,
        "targets": example["answer"]
    }

def formatting_prompts_func(example):
    return {
        "text": example["inputs"] + "### Response: " + example["targets"]
    }

def parse_final_answer(text):
    marker = "####"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    after = text[idx+len(marker):].strip()
    tokens = after.split()
    if not tokens:
        return None
    try:
        val = float(tokens[0].strip(",.!?"))
        return int(val) if val.is_integer() else val
    except ValueError:
        return None

############################################
# Main Function
############################################
def main():
    global PROMPT_NO_INPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["first", "second", "third"], default="first")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--full_parameter", action="store_true")
    parser.add_argument("--no_prompt_template", action="store_true")
    parser.add_argument("--universal_prompt", action="store_true")
    parser.add_argument("--adapter_type", choices=["lora", "ia3"], default="lora")
    args = parser.parse_args()

    # Prompt handling
    if args.no_prompt_template:
        PROMPT_NO_INPUT = "### Instruction:\n{instruction}\n\n"
    elif args.universal_prompt:
        PROMPT_NO_INPUT = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
        )

    # Output path
    model_id_safe = args.model_name.replace("/", "-")
    suffix = f"{args.num_samples or args.split}-{args.adapter_type if not args.full_parameter else 'full-parameter'}-{args.batch_size}-{args.learning_rate}"
    if args.no_prompt_template:
        suffix += "-no-prompt-template"
    if args.universal_prompt:
        suffix += "-same-prompt-template"
    if args.num_epochs != 3:
        suffix += f"-num-epochs-{args.num_epochs}"
    output_dir = f"./{model_id_safe}-gsm8k-{suffix}"

    # Load dataset
    dataset = load_dataset("gsm8k", "main")
    n_total = len(dataset["train"])
    split_at = int(0.8 * n_total)

    if args.num_samples:
        avail = list(range(split_at, n_total))
        chosen = random.sample(avail, min(args.num_samples, len(avail)))
        train_dataset = dataset["train"].select(chosen)
    else:
        third = split_at // 3
        bounds = {"first": (0, third), "second": (third, 2 * third), "third": (2 * third, split_at)}
        start, end = bounds[args.split]
        train_dataset = dataset["train"].select(range(start, end))

    test_dataset = dataset["test"].select(range(max(1, int(0.3 * len(dataset["test"])))))

    # Preprocess
    train_dataset = train_dataset.map(preprocess_gsm8k).map(formatting_prompts_func)
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "text"])

    test_dataset = test_dataset.map(preprocess_gsm8k).map(formatting_prompts_func)
    test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col != "text"])

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    if args.full_parameter:
        model = base_model
    elif args.adapter_type == "lora":
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_cfg)
    else:
        ia3_cfg = IA3Config(
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, ia3_cfg)

    response_template = "### Response:"
    response_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_ids,
        tokenizer=tokenizer,
        padding_free=False
    )

    def compute_metrics(eval_preds):
        logits = eval_preds.predictions
        labels = eval_preds.label_ids
        if logits is None or labels is None:
            return {}
        predictions = np.argmax(logits, axis=-1)
        pred_vals, true_vals = [], []

        for pred_ids, true_ids in zip(predictions, labels):
            pred_text = tokenizer.decode([x for x in pred_ids if x != -100], skip_special_tokens=True).strip()
            true_text = tokenizer.decode([y for y in true_ids if y != -100], skip_special_tokens=True).strip()
            pred_val = parse_final_answer(pred_text)
            true_val = parse_final_answer(true_text)

            pred_vals.append(pred_val if pred_val is not None else -100000)
            true_vals.append(true_val)

        correct = sum(p == t for p, t in zip(pred_vals, true_vals))
        accuracy = correct / len(true_vals) if true_vals else 0.0
        return {"accuracy": accuracy}

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        optim="paged_adamw_32bit",
        eval_strategy="no",
        logging_steps=10,
        warmup_steps=10,
        save_strategy="epoch",
        report_to=None,  # Disable wandb
        push_to_hub=False,  # Make it anonymous
        eval_accumulation_steps=4,
        packing=False,
        eval_packing=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logger.info("Starting fine-tuning on GSM8K ...")
    trainer.train()
    logger.info(f"Training completed. Model saved to {output_dir}.")

if __name__ == "__main__":
    main()
