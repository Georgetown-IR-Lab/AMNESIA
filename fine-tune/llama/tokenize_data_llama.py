# Adapted From: https://raw.githubusercontent.com/zphang/minimal-llama/main/tokenize_dataset.py
# Modified for LLaMA 3 8B

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
        "--model_name", type=str, default="meta-llama/Meta-Llama-3-8B"
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

    # Concatenate all tokens with BOS/EOS separators (same as original)
    all_tokens = [tokenizer.bos_token_id] + [
        tok
        for row in all_tokenized
        for tok in row + [tokenizer.eos_token_id, tokenizer.bos_token_id]
    ]

    # Chunk into fixed-length sequences
    seq_len = args.max_length
    truncated_tokens = all_tokens[: (len(all_tokens) // seq_len) * seq_len]
    arr = np.array(truncated_tokens).reshape(-1, seq_len)
    ds = datasets.Dataset.from_dict({"input_ids": arr})

    ds.save_to_disk(args.save_path)
    print(f"Generated {arr.shape[0]} samples of length {seq_len}.")


if __name__ == "__main__":
    main()
