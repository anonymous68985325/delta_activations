import argparse
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType

parser = argparse.ArgumentParser(description="DPO fine-tuning for Llama-3.1-8B-Instruct")
parser.add_argument("--split", choices=["first", "second", "third"], default="first")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--num_samples", type=int, default=None)
parser.add_argument("--learning_rate", type=float, default=5e-6)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--gpu_id", type=str, default=None)
parser.add_argument("--dpo_dataset", type=str, default="HumanLLMs/Human-Like-DPO-Dataset", choices=[
    "HumanLLMs/Human-Like-DPO-Dataset",
    "trl-lib/ultrafeedback_binarized",
    "AI4Chem/ChemPref-DPO-for-Chemistry-data-en",
    "abacusai/MetaMath_DPO_FewShot"
])

def main(cli_args=None):
    args = parser.parse_args(args=cli_args)

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Configuration:")
    print(f"- Model:         {args.model_name}")
    print(f"- Dataset:       {args.dpo_dataset}")
    print(f"- Split:         {args.split}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Batch size:    {args.batch_size}")
    print(f"- Num samples:   {args.num_samples}")
    print(f"- Device:        {device}")

    full_dataset = load_dataset(path=args.dpo_dataset, split="train")
    n_total = len(full_dataset)
    print(f"Full dataset size: {n_total}")

    if "HumanLLMs" in args.dpo_dataset:
        split_at = int(0.85 * n_total)
    elif "Chem" in args.dpo_dataset:
        split_at = int(0.98 * n_total)
    elif "Math" in args.dpo_dataset:
        split_at = int(0.025 * n_total)
    else:
        split_at = int(0.15 * n_total)

    print(f"Split at: {split_at} ({split_at / n_total:.0%})")

    if args.num_samples is not None:
        avail = list(range(split_at, n_total))
        chosen = random.sample(avail, min(args.num_samples, len(avail)))
        train_dataset = full_dataset.select(chosen)
        print(f"Selected {len(chosen)} random examples for training")
    else:
        region = split_at
        third = region // 3
        if args.split == "first":
            start_idx, end_idx = 0, third
        elif args.split == "second":
            start_idx, end_idx = third, 2 * third
        else:
            start_idx, end_idx = 2 * third, region
        train_dataset = full_dataset.select(range(start_idx, end_idx))
        print(f"Using {args.split} split: {start_idx}–{end_idx} ({len(train_dataset)})")

    eval_dataset = full_dataset.select(range(split_at, min(split_at + 500, n_total)))
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    if "Chem" in args.dpo_dataset:
        def rename_instruction(example):
            example["prompt"] = example["instruction"]
            return example
        train_dataset = train_dataset.map(rename_instruction)
        eval_dataset = eval_dataset.map(rename_instruction)

    print("\nDataset sample:")
    for k, v in train_dataset[0].items():
        preview = v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v
        print(f"{k}: {preview}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_id_safe = args.model_name.replace("/", "-")
    dataset_id_safe = args.dpo_dataset.replace("/", "-")
    split_tag = str(args.num_samples) if args.num_samples else args.split
    out_dir = f"./dpo-{model_id_safe}-{dataset_id_safe}-{split_tag}-{args.batch_size}-{args.learning_rate}"

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    training_args = DPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=-1,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        warmup_ratio=0.1,
        bf16=device == "cuda",
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        use_mps_device=device == "mps",
        hub_model_id=None,  # Hub upload disabled
        beta=0.1,
        max_prompt_length=8500 if "Math" in args.dpo_dataset else 1024,
        max_length=14000 if "Math" in args.dpo_dataset else 1536,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(out_dir)
    print(f"✅ Model saved locally at: {out_dir}")

if __name__ == "__main__":
    main()
