import logging
import os
import random
import numpy as np
import torch
from typing import Dict
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, IA3Config
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Below is a medical question based on a PubMed article. "
    "Please answer it briefly based on the provided context.\n\n"
    "### Question:\n{question}\n\n"
    "### Context:\n{context}\n\n"
)

LORA_RANK = 8

def preprocess_pubmedqa(example):
    question = example["question"]
    context = " ".join(example["context"]["contexts"])  
    long_answer = example["long_answer"].strip()
    prompt_str = PROMPT_TEMPLATE.format(question=question, context=context)
    return {"inputs": prompt_str, "targets": long_answer}

def formatting_prompts_func(example):
    return {"text": example["inputs"] + "### Response: " + example["targets"]}

def main():
    global PROMPT_TEMPLATE

    import argparse
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

    if args.no_prompt_template:
        PROMPT_TEMPLATE = (
            "### Question:\n{question}\n\n"
            "### Context:\n{context}\n\n"
        )
    elif args.universal_prompt:
        PROMPT_TEMPLATE = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{question} Input: {context}\n\n"
        )

    model_id_safe = args.model_name.replace("/", "-")
    suffix = f"{args.num_samples or args.split}-{args.adapter_type if not args.full_parameter else 'full-parameter'}-{args.batch_size}-{args.learning_rate}"
    if args.no_prompt_template:
        suffix += "-no-prompt-template"
    if args.universal_prompt:
        suffix += "-same-prompt-template"
    if args.num_epochs != 3:
        suffix += f"-num-epochs-{args.num_epochs}"
    output_dir = f"./{model_id_safe}-pubmedqa-{suffix}"

    split_part = args.split
    set_num_map = {"first": 1, "second": 2, "third": 3}
    args.set_num = set_num_map[split_part]

    logger.info("Loading PubMedQA...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    train_dataset = dataset["train"]
    
    n_total = len(train_dataset)
    split_at = int(0.1 * n_total)
    logger.info(f"Total examples: {n_total} | Split at: {split_at}")

    if args.num_samples:
        avail = list(range(split_at, n_total))
        chosen = random.sample(avail, min(args.num_samples, len(avail)))
        train_dataset = train_dataset.select(chosen)
    else:
        region = split_at
        third = region // 3
        bounds = {"first": (0, third), "second": (third, 2 * third), "third": (2 * third, region)}
        start_idx, end_idx = bounds[split_part]
        train_dataset = train_dataset.select(range(start_idx, end_idx))

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info("Preprocessing train set ...")
    train_dataset = train_dataset.map(preprocess_pubmedqa).map(formatting_prompts_func)
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col != "text"]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    if args.full_parameter:
        model = base_model
    elif args.adapter_type == "lora":
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=16,
            target_modules=["k_proj", "q_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
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
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, ia3_cfg)

    response_template_ids = [tokenizer.encode("### Response:", add_special_tokens=False)[1]]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=response_template_ids,
        tokenizer=tokenizer,
        padding_free=False
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        optim="paged_adamw_32bit",
        eval_strategy="no",
        logging_steps=10,
        warmup_steps=10,
        save_strategy="epoch",
        report_to=None,
        push_to_hub=False,
        packing=False,
        eval_packing=False,
        max_seq_length=768,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    logger.info("Starting fine-tuning on PubMedQA …")
    trainer.train()
    logger.info(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
