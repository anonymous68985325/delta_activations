import logging
import os
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

LORA_RANK = 8

# Prompt templates
HELLASWAG_TEMPLATES = [
    "What happens next in this paragraph?\n\n{context}\n\n{options_}\n\nWrite your answer and conclude with the correct letter after '###'.",
    "Continue writing the next sentence:\n\n{context}\n\n{options_}\n\nAnswer and then give the letter after '###'.",
    "Select from options:\n\n{context}\n\n{options_}\n\nWrite the best ending and then provide the answer letter after '###'.",
    "Complete the next sentence:\n\n{context}\n\n{options_}\n\nGive your full answer and final letter below '###'.",
    "Write the next sentence:\n\n{context}\n\n{options_}\n\nInclude the best ending and then label it after '###'.",
    "How does the paragraph end?\n\n{context}\n\n{options_}\n\nWrite your choice and show the answer letter after '###'.",
    "{options_}\nChoose from options above:\n\n{context}\n\nWrite your answer and then give the letter after '###'.",
    "What happens next?\n\n{context}\n\n{options_}\n\nAnswer and write the correct letter after '###'.",
    "What is the most logical next event?\n\n{context}\n\n{options_}\n\nRespond and then show the correct choice letter after '###'.",
    "Write the next sentence in the story:\n\n{context}\n\n{options_}\n\nInclude your answer and the letter after '###'."
]

def preprocess_hellaswag(example: Dict[str, str]) -> Dict[str, str]:
    context = example["ctx"]
    endings = example["endings"]
    label = int(example["label"])
    options_ = "\n".join([f"{chr(65+i)}. {endings[i]}" for i in range(4)])
    prompt_template = random.choice(HELLASWAG_TEMPLATES)
    prompt_str = prompt_template.format(context=context, options_=options_)
    correct_option = ["A", "B", "C", "D"][label]
    correct_ending = endings[label]
    answer_str = f"\n\nThe correct answer is: {correct_ending}\n###\n{correct_option}"
    return {"inputs": prompt_str, "targets": answer_str}

def formatting_prompts_func(example):
    return {"text": example["inputs"] + "### Response: " + example["targets"]}

def parse_correct_answer_letter(text):
    lines = text.strip().splitlines()
    for i in range(len(lines)):
        if lines[i].strip() == "###" and i + 1 < len(lines):
            candidate = lines[i + 1].strip().strip(".!?,")
            if candidate in ["A", "B", "C", "D"]:
                return candidate
    return None

def main():
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
    parser.add_argument("--adapter_type", type=str, choices=["lora", "ia3"], default="lora")
    args = parser.parse_args()

    if args.no_prompt_template:
        global HELLASWAG_TEMPLATES
        HELLASWAG_TEMPLATES = [
            "{context}\n\n{options_}\n\nInclude your answer and the letter after '###'."
        ]
    elif args.universal_prompt:
        HELLASWAG_TEMPLATES = [
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{context} Input: {options_}\n\n"
        ]

    set_name_map = {"first": 1, "second": 2, "third": 3}
    set_num = set_name_map[args.split]

    model_id_safe = args.model_name.replace("/", "-")
    suffix = f"{args.num_samples or args.split}-{args.adapter_type if not args.full_parameter else 'full-parameter'}-{args.batch_size}-{args.learning_rate}"
    if args.no_prompt_template:
        suffix += "-no-prompt-template"
    if args.universal_prompt:
        suffix += "-same-prompt-template"
    if args.num_epochs != 3:
        suffix += f"-num-epochs-{args.num_epochs}"
    output_dir = f"./{model_id_safe}-hellaswag-{suffix}"

    dataset = load_dataset("Rowan/hellaswag")
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"].select(range(1000))

    logger.info("Preprocessing dataset...")
    train_dataset = train_dataset.map(preprocess_hellaswag).map(formatting_prompts_func)
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != "text"])

    if args.num_samples:
        chosen = random.sample(range(len(train_dataset)), min(args.num_samples, len(train_dataset)))
        train_dataset = train_dataset.select(chosen)
    else:
        third = len(train_dataset) // 3
        bounds = [(0, third), (third, 2 * third), (2 * third, len(train_dataset))][set_num - 1]
        train_dataset = train_dataset.select(range(*bounds))

    test_dataset = test_dataset.map(preprocess_hellaswag).map(formatting_prompts_func)
    test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col != "text"])

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
        predictions = np.argmax(eval_preds.predictions, axis=-1)
        pred_vals, true_vals = [], []

        for pred_ids, label_ids in zip(predictions, eval_preds.label_ids):
            pred_text = tokenizer.decode([i for i in pred_ids if i != -100], skip_special_tokens=True).strip()
            true_label = parse_correct_answer_letter(pred_text)
            correct_label = ["A", "B", "C", "D"][test_dataset[pred_vals.__len__()]["label"]]
            if true_label is not None and correct_label is not None:
                pred_vals.append(true_label)
                true_vals.append(correct_label)

        accuracy = sum(p == t for p, t in zip(pred_vals, true_vals)) / len(true_vals) if true_vals else 0.0
        logger.info(f"Eval => accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy}

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
        eval_packing=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning on HellaSwag ...")
    trainer.train()
    logger.info(f"Training completed. Model saved to {output_dir}.")

if __name__ == "__main__":
    main()
