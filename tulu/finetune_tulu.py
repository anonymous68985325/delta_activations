import argparse
import logging
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# ----------------- Config -----------------
NUM_EPOCHS = 3
LORA_RANK = 8
DTYPE = torch.bfloat16

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ----------------- Helpers -----------------
def keep_until_first_assistant(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    kept = []
    for m in msgs:
        kept.append({"role": m["role"], "content": m["content"]})
        if m["role"] == "assistant":
            break
    return kept

def preprocess_tulu(example, tokenizer):
    msgs = keep_until_first_assistant(example["messages"])
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=[
        "science", "cot", "sharegpt", "wizardlm", "hard_coded", "open_orca",
        "lima", "science.scifact_json", "science.evidence_inference",
        "science.scierc_relation", "science.scierc_ner",
        "science.scitldr_aic", "science.qasper_truncated_4000",
        "flan_v2", "code_alpaca", "oasst1", "gpt4_alpaca"
    ])
    parser.add_argument("--split", choices=["first", "second", "third"], default="first")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_seq_len", type=float, default=512)
    parser.add_argument("--full_parameter", action="store_true")
    args = parser.parse_args()

    SCIENCE_SPLITS = [
        "science.scifact_json", "science.scierc_ner", "science.qasper_truncated_4000",
        "science.evidence_inference", "science.scierc_relation", "science.scitldr_aic"
    ]
    keep_sets = set(SCIENCE_SPLITS) if args.subset == "science" else {args.subset}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    # Inject template if needed
    if tokenizer.chat_template is None and "llama" in args.model_name.lower():
        tokenizer.chat_template = (
            "{% for message in messages %}{{ bos_token if loop.index0 == 0 else '' }}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content']|trim }}<|eot_id|>{% endfor %}"
        )
    elif tokenizer.chat_template is None and "gemma" in args.model_name.lower():
        tokenizer.chat_template = (
            "{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}"
        )

    if "gemma" not in args.model_name.lower():
        assist_header = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}],
            tokenize=False, add_generation_prompt=False
        ).split(tokenizer.eos_token)[0]
        response_template_ids = tokenizer.encode(assist_header, add_special_tokens=False)[1:-2]
    elif "qwen" in args.model_name.lower():
        response_template_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[1:]
    else:
        response_template_ids = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)[1:]

    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer, padding_free=False
    )

    log.info("Loading tulu-v2-sft-mixture …")
    raw = load_dataset("allenai/tulu-v2-sft-mixture", split="train")
    raw = raw.filter(lambda ex: ex["dataset"] in keep_sets)

    processed = raw.map(lambda ex: preprocess_tulu(ex, tokenizer), remove_columns=raw.column_names)
    processed = processed.filter(lambda ex: ex is not None)

    n = len(processed)
    log.info(f"Total examples kept: {n}")

    k = min(n, 10_000) if n >= 10_000 else int(0.98 * n)

    if args.num_samples:
        idx = random.sample(range(k, n), min(args.num_samples, n - k))
        train = processed.select(idx)
    else:
        third = k // 3
        if args.split == "first":
            train = processed.select(range(0, third))
        elif args.split == "second":
            train = processed.select(range(third, 2 * third))
        else:
            train = processed.select(range(2 * third, k))
    test = processed.select(range(k, n))

    log.info(f"Train {len(train)} | Test {len(test)}")

    core = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=DTYPE, device_map="auto")

    if args.full_parameter:
        model = core
        suffix = "full"
    else:
        lora_cfg = LoraConfig(
            r=LORA_RANK,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "gate_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(core, lora_cfg)
        suffix = "lora"

    subset_safe = args.subset.replace(".", "-")
    model_safe = args.model_name.replace("/", "-")
    output_dir = f"./{model_safe}-tulu-{subset_safe}-{args.split}-{suffix}-{args.batch_size}-{args.learning_rate}"

    config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=NUM_EPOCHS,
        optim="paged_adamw_32bit",
        learning_rate=args.learning_rate,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=10,
        report_to=None,
        push_to_hub=False,
        max_seq_length=args.max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=test,
        data_collator=collator,
    )

    log.info("🚀 Starting fine-tuning …")
    trainer.train()
    log.info(f"✅ Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
