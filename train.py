#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train StarCoderBase-3B (4-bit QLoRA) to map:
  [Source code + Target sketch(with <pad> holes)]  -->  [Full Target code]

Input file: java_to_rust_pair.jsonl
Each line: {"src_uid": "...", "source_lang": "Java", "target_lang": "Rust",
            "source_code": "...", "target_code": "..."}

Requires:
  pip install -U transformers accelerate bitsandbytes peft datasets
"""
import os, json, math, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ----------------------------- Masking utils ---------------------------------
def mask_tokens_with_pad(
    tokenizer,
    text: str,
    mask_frac: float = 0.30,
    min_span: int = 3,
    max_span: int = 15,
    pad_token_literal: str = "<pad>",
) -> str:
    """
    Turn target code into a 'sketch' by masking random token spans with the literal `<pad>`.

    - Works at tokenizer-token level to keep masks roughly syntax-aware.
    - Does NOT use tokenizer.pad_token_id; we insert the string `<pad>` in the text so the model
      learns to read sketches with holes.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    n = len(ids)
    if n == 0:
        return text

    # Desired number of tokens to mask (approximately)
    tokens_to_mask = max(1, int(mask_frac * n))

    masked = [False] * n
    total = 0
    # Greedy sample of non-overlapping spans
    while total < tokens_to_mask:
        span_len = random.randint(min_span, max_span)
        start = random.randint(0, n - 1)
        end = min(n, start + span_len)
        if any(masked[start:end]):
            continue
        for i in range(start, end):
            masked[i] = True
        total += (end - start)

    # Build mixed token list: unmasked tokens as-is; masked spans collapsed to one `<pad>`
    pieces: List[str] = []
    i = 0
    while i < n:
        if not masked[i]:
            pieces.append(tokenizer.convert_ids_to_tokens(ids[i]))
            i += 1
        else:
            # collapse a contiguous masked block to one <pad>
            j = i
            while j < n and masked[j]:
                j += 1
            pieces.append(pad_token_literal)
            i = j

    # Convert token pieces back to text (keeping `<pad>` untouched)
    # convert_tokens_to_string will join non-special tokens; we then place '<pad>' verbatim.
    text_out_parts: List[str] = []
    tmp_buffer: List[str] = []
    for tok in pieces:
        if tok == pad_token_literal:
            if tmp_buffer:
                text_out_parts.append(tokenizer.convert_tokens_to_string(tmp_buffer))
                tmp_buffer = []
            text_out_parts.append(pad_token_literal)
        else:
            tmp_buffer.append(tok)
    if tmp_buffer:
        text_out_parts.append(tokenizer.convert_tokens_to_string(tmp_buffer))
    return "".join(text_out_parts)


# ------------------------------ Dataset --------------------------------------
class SketchFillerDataset(Dataset):
    """
    Builds training examples on-the-fly:

    Input (context):  optional prompt + Source code + Target sketch
    Target labels:    full Target code (only these tokens contribute to loss)
    """
    def __init__(
        self,
        hf_dataset: HFDataset,
        tokenizer: AutoTokenizer,
        max_len: int,
        mask_frac: float,
        source_key: str = "source_code",
        target_key: str = "target_code",
        src_lang_key: str = "source_lang",
        tgt_lang_key: str = "target_lang",
        prompt_key: str = None,  # optional in data
    ):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.max_len = max_len
        self.mask_frac = mask_frac
        self.source_key = source_key
        self.target_key = target_key
        self.src_lang_key = src_lang_key
        self.tgt_lang_key = tgt_lang_key
        self.prompt_key = prompt_key

    def __len__(self):
        return len(self.ds)

    def _build_strings(self, row: dict) -> Tuple[str, str]:
        src_lang = row.get(self.src_lang_key, "Source")
        tgt_lang = row.get(self.tgt_lang_key, "Target")
        src = row[self.source_key]
        tgt = row[self.target_key]

        # Make a sketch by masking spans in target code with the literal `<pad>`
        sketch = mask_tokens_with_pad(self.tok, tgt, mask_frac=self.mask_frac)

        # Optional task/UTs prompt
        nl = (row.get(self.prompt_key) or "").strip() if self.prompt_key else ""

        context = ""
        if nl:
            context += nl + "\n\n"
        context += f"# Source ({src_lang}):\n{src}\n\n"
        context += f"# Target ({tgt_lang}) Sketch:\n{sketch}\n\n"
        context += f"# Target ({tgt_lang}) Code:\n"

        target_text = tgt
        return context, target_text

    def __getitem__(self, idx: int):
        row = self.ds[int(idx)]
        context, tgt = self._build_strings(row)

        # Tokenize separately to know where to mask labels
        ctx_ids = self.tok(context, add_special_tokens=False).input_ids
        tgt_ids = self.tok(tgt, add_special_tokens=False).input_ids

        input_ids = ctx_ids + tgt_ids
        if len(input_ids) > self.max_len:
            # Keep the tail (usually includes 'Code:' + target)
            input_ids = input_ids[-self.max_len:]

        attn = [1] * len(input_ids)

        # Determine how many of the kept tokens belong to the context
        ctx_kept = max(0, len(ctx_ids) - max(0, len(ctx_ids) + len(tgt_ids) - self.max_len))

        # Labels: ignore context (and sketch) part
        labels = ([-100] * ctx_kept) + input_ids[ctx_kept:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------------------ Main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="/home/jalend/code-translation/data/utils/java_to_rust_pairs.jsonl", help="Path to JSONL data file")
    ap.add_argument("--output_dir", default="chkpts/scb3b_filler_sketch_qlora", help="Where to save adapters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--mask_frac", type=float, default=0.30, help="Fraction of target tokens to mask into <pad> for sketch")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--eval_ratio", type=float, default=0.05, help="Hold out 5%% by default for eval")
    ap.add_argument("--hf_home", default=None)
    ap.add_argument("--hf_cache", default=None)
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    if args.hf_cache:
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    set_seed(args.seed)

    checkpoint = "bigcode/starcoderbase-3b"

    # --- 4-bit quantization (NF4) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    # Keep pad_token for padding mechanics as EOS; teach the literal '<pad>' as an extra special string
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"]})

    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    if added:
        base_model.resize_token_embeddings(len(tokenizer))

    base_model = prepare_model_for_kbit_training(base_model)

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_cfg)
    model.print_trainable_parameters()

    # --- Load JSONL as HF dataset, split train/val ---
    # We construct a tiny HF dataset from the local JSONL
    def gen():
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    raw = list(gen())
    n = len(raw)
    assert n > 0, "Empty dataset."

    random.Random(args.seed).shuffle(raw)
    n_eval = max(1, int(args.eval_ratio * n))
    eval_data = raw[:n_eval]
    train_data = raw[n_eval:]

    hf_train = HFDataset.from_list(train_data)
    hf_eval  = HFDataset.from_list(eval_data)

    train_ds = SketchFillerDataset(
        hf_dataset=hf_train,
        tokenizer=tokenizer,
        max_len=args.max_len,
        mask_frac=args.mask_frac,
    )
    eval_ds = SketchFillerDataset(
        hf_dataset=hf_eval,
        tokenizer=tokenizer,
        max_len=args.max_len,
        mask_frac=args.mask_frac,  # keep same masking at eval so model sees realistic sketches
    )

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)       # saves LoRA adapters
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapters to: {args.output_dir}")


if __name__ == "__main__":
    main()
