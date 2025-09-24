import os
import logging
import argparse
import random
import numpy as np
import torch
from typing import Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, IA3Config
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_NO_INPUT = (
    "Below is a legal‑reasoning task from the LegalBench benchmark. "
    "You must determine if the clause contains an answer to the question, and classify the question-clause pair as Relevant or Irrelevant. State your answer in the following format: 'The correct answer is relevant/irrelevant'.\n\n"
    "\n{instruction}\n\n"
)

LORA_RANK = 8

def preprocess_legalbench(example: Dict[str, str]) -> Dict[str, str]:
    inst = "Question: " + example["question"] + " Clause: " + example["text"]
    target = "The correct answer is " + example["answer"].lower().strip() + "."
    prompt = PROMPT_NO_INPUT.format(instruction=inst)
    return {"inputs": prompt, "targets": target}

def formatting_prompts_func(example):
    return {"text": example["inputs"] + "### Response: " + example["targets"]}

def main():
    global PROMPT_NO_INPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--legal_task", default="privacy_policy_qa")
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

    if args.no_prompt_template:
        PROMPT_NO_INPUT = "\n{instruction}\n\n"
    elif args.universal_prompt:
        PROMPT_NO_INPUT = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
        )

    model_id_safe = args.model_name.replace("/", "-")
    suffix = f"{args.num_samples or args.split}-{args.adapter_type if not args.full_parameter else 'full-parameter'}-{args.batch_size}-{args.learning_rate}"
    if args.no_prompt_template:
        suffix += "-no-prompt-template"
    if args.universal_prompt:
        suffix += "-same-prompt-template"
    if args.num_epochs != 3:
        suffix += f"-num-epochs-{args.num_epochs}"
    output_dir = f"./{model_id_safe}-legalbench-{suffix}"

    logger.info(f"Loading LegalBench task: {args.legal_task}")
    ds = load_dataset("nguha/legalbench", args.legal_task, trust_remote_code=True)
    full_ds = ds["test"]  # LegalBench has only a public test set

    full_ds = full_ds.map(preprocess_legalbench).map(formatting_prompts_func)
    full_ds = full_ds.remove_columns([c for c in full_ds.column_names if c != "text"])

    n_total = len(full_ds)
    split_at = int(0.8 * n_total)
    logger.info(f"Total examples: {n_total}, split_at={split_at}")

    if args.num_samples:
        avail = list(range(split_at, n_total))
        chosen = random.sample(avail, min(args.num_samples, len(avail)))
        train_ds = full_ds.select(chosen)
    else:
        third = split_at // 3
        bounds = {"first": (0, third), "second": (third, 2 * third), "third": (2 * third, split_at)}
        start_idx, end_idx = bounds[args.split]
        train_ds = full_ds.select(range(start_idx, end_idx))

    test_ds = full_ds.select(range(split_at, n_total))
    print(f"Using split '{args.split}': train={len(train_ds)}, test={len(test_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    if args.full_parameter:
        model = base_model
    elif args.adapter_type == "lora":
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=False,
        )
        model = get_peft_model(base_model, lora_cfg)
    else:
        ia3_cfg = IA3Config(
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, ia3_cfg)

    response_ids = tokenizer.encode("### Response:", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids=response_ids, tokenizer=tokenizer, padding_free=False)

    config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        optim="paged_adamw_32bit",
        logging_steps=10,
        eval_strategy="no",
        save_strategy="epoch",
        warmup_steps=10,
        learning_rate=args.learning_rate,
        report_to=None,
        push_to_hub=False,
        eval_accumulation_steps=4,
        packing=False,
        eval_packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
    )

    logger.info("Starting fine-tuning on LegalBench …")
    trainer.train()
    logger.info(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
