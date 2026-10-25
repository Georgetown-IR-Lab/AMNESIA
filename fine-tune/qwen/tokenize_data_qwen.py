# Adapted From: https://raw.githubusercontent.com/zphang/minimal-llama/main/tokenize_dataset.py
# Modified for Qwen3-8B-Base

import argparse
import json

import datasets
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from transformers import AutoTokenizer

from utils import modify_special_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to .jsonl file")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-8B-Base"
    )
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--text_field", type=str, default="patient_note",
                        help="JSON field containing text for pre-training")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Sequence length for chunking")
    parser.add_argument("--num_workers", type=int, default=7,
                        help="Number of parallel workers")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = modify_special_tokens(tokenizer)

    # Load JSONL and deduplicate by text field
    print(f"Loading data from {args.input_path}...")
    records = []
    seen = set()
    with open(args.input_path, "r") as f:
        for line in f:
            row = json.loads(line)
            text = row.get(args.text_field, "").strip()
            if text and text not in seen:
                seen.add(text)
                records.append(text)

    print(f"Found {len(records)} unique texts (deduplicated)")
    df = pd.DataFrame({"text": records}).sample(frac=1, random_state=42)

    # Tokenize all texts
    print("Tokenizing...")
    all_tokenized = []
    for text in tqdm.tqdm(df["text"], total=len(df)):
        all_tokenized.append(tokenizer.encode(text, add_special_tokens=False))

    total_tokens = sum(len(t) for t in all_tokenized)
    print(f"Total number of tokens is {total_tokens:,}")

    # ─── Document separation ────────────────────────────────────────
    # IMPORTANT DIFFERENCE FROM LLaMA:
    # Qwen3 has NO BOS token (tokenizer.bos_token_id is None), so the original
    # "[BOS] doc [EOS] [BOS] doc [EOS] ..." packing scheme cannot be used --
    # splicing None into the token list would corrupt the array.
    #
    # For Qwen3-8B-Base, "<|endoftext|>" (id 151643) IS the model's eos_token
    # and is the end-of-text token used during pretraining, so it is exactly
    # the right delimiter between documents for continued pre-training. We
    # resolve it explicitly rather than assuming tokenizer.eos_token_id, so the
    # script stays correct even if pointed at the instruct model (whose eos is
    # the ChatML "<|im_end|>", which is NOT a raw-text document boundary).
    eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert eot_id is not None and eot_id >= 0, "Could not resolve <|endoftext|>"

    all_tokens = []
    for row in all_tokenized:
        all_tokens.extend(row)
        all_tokens.append(eot_id)

    # Chunk into fixed-length sequences
    seq_len = args.max_length
    truncated_tokens = all_tokens[: (len(all_tokens) // seq_len) * seq_len]
    arr = np.array(truncated_tokens).reshape(-1, seq_len)
    ds = datasets.Dataset.from_dict({"input_ids": arr})

    ds.save_to_disk(args.save_path)
    print(f"Generated {arr.shape[0]} samples of length {seq_len}.")


if __name__ == "__main__":
    main()
