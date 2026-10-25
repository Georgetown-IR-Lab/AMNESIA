#!/usr/bin/env python3
"""Large-scale batched generation for patient QA with rank-sharding and resume.

Adapted from open-unlearning inference script. Adds:
  - Data sharding across torchrun/srun-launched processes via LOCAL_RANK/WORLD_SIZE
  - Per-rank output file so append writes are race-free: <output>.rank{R}.jsonl
  - Per-process single-GPU placement via device_map={"": f"cuda:{local_rank}"}

Input JSONL records must contain at least: id, question, patient_note, answer.
Each result row is: {id, question, reference_answer, prediction,
                     patient_id, category, input_tokens, output_tokens, time_seconds}.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from patient_clinical_prompt import build_inference_prompt  # noqa: E402


_PATIENT_ID_RE = re.compile(r"^([0-9]+-[0-9]+)")


def _patient_id(row_id: str) -> str:
    m = _PATIENT_ID_RE.match(str(row_id))
    return m.group(1) if m else str(row_id)


def _get_rank_world_size():
    """Pull rank/world size from torchrun or srun env vars; default to single-process."""
    rank = int(
        os.environ.get("RANK")
        or os.environ.get("SLURM_PROCID")
        or 0
    )
    world_size = int(
        os.environ.get("WORLD_SIZE")
        or os.environ.get("SLURM_NTASKS")
        or 1
    )
    local_rank = int(
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or rank
    )
    return rank, world_size, local_rank


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


def count_jsonl_rows(path: str) -> int:
    with open(path, "r") as f:
        return sum(1 for line in f if line.strip())


def load_all_done_ids(output_file: str) -> set:
    """IDs already generated in merged output and/or rank shards."""
    done = load_checkpoint(output_file)
    for path in sorted(glob.glob(f"{output_file}.rank*.jsonl")):
        done |= load_checkpoint(path)
    return done


def generation_complete(output_file: str, test_file: str) -> bool:
    expected = count_jsonl_rows(test_file)
    return len(load_all_done_ids(output_file)) >= expected


def merge_generation_shards(output_file: str) -> int:
    """Merge merged output + rank shards into output_file; last write wins on id.

    Deduplicates by ``id`` (never fails the job for duplicates).
    """
    rows_by_id = {}
    n_raw = 0
    sources = []
    if os.path.isfile(output_file):
        sources.append(output_file)
    sources.extend(sorted(glob.glob(f"{output_file}.rank*.jsonl")))
    for path in sources:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rid = obj.get("id")
                if rid is None or rid == "":
                    continue
                n_raw += 1
                rows_by_id[rid] = obj
    rows = sorted(rows_by_id.values(), key=lambda r: r["id"])
    with open(output_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for path in sorted(glob.glob(f"{output_file}.rank*.jsonl")):
        os.remove(path)
    n_unique = len(rows)
    n_dup = n_raw - n_unique
    if n_dup > 0:
        print(
            f"[merge] {n_unique} unique ids "
            f"(dropped {n_dup} duplicate rows) -> {output_file}"
        )
    else:
        print(f"[merge] {n_unique} rows -> {output_file}")
    return n_unique


def dedupe_jsonl_by_id(path: str) -> int:
    """Rewrite JSONL in place with one row per id (last wins). Returns unique count."""
    if not os.path.isfile(path):
        return 0
    rows_by_id = {}
    n_raw = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if rid is None or rid == "":
                continue
            n_raw += 1
            rows_by_id[rid] = obj
    rows = sorted(rows_by_id.values(), key=lambda r: r["id"])
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_unique = len(rows)
    n_dup = n_raw - n_unique
    if n_dup > 0:
        print(f"[dedupe] {path}: kept {n_unique}, dropped {n_dup} duplicates")
    return n_unique


def run_batch(model, tokenizer, batch, args):
    import torch

    prompts = [
        build_inference_prompt(item, prompt_mode=args.prompt_mode) for item in batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    results = []
    input_len = inputs["input_ids"].shape[1]
    for i, item in enumerate(batch):
        actual_input_len = (
            (inputs["input_ids"][i] != tokenizer.pad_token_id).sum().item()
        )
        generated_ids = outputs[i][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        results.append(
            {
                "id": item["id"],
                "patient_id": _patient_id(item["id"]),
                "split": args.split_name,
                "category": item.get("category"),
                "question": item["question"],
                "reference_answer": item.get("answer", ""),
                "prediction": response,
                "input_tokens": actual_input_len,
                "output_tokens": len(generated_ids),
                "time_seconds": round(elapsed / max(1, len(batch)), 3),
            }
        )
    return results, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--output_file", required=True,
                    help="Base output path; actual write goes to <output_file>.rank{R}.jsonl")
    ap.add_argument("--split_name", default="eval")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--do_sample", action="store_true", default=False)
    ap.add_argument(
        "--prompt_mode",
        default="finetune",
        choices=("finetune", "bare_qa"),
        help="finetune=note+question template; bare_qa/question only (closed-book)",
    )
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument(
        "--check_complete",
        action="store_true",
        help="Exit 0 if all test rows are generated (merged + shards), else 1.",
    )
    ap.add_argument(
        "--merge_only",
        action="store_true",
        help="Merge merged output + rank shards into output_file and exit.",
    )
    args = ap.parse_args()

    if args.merge_only:
        merge_generation_shards(args.output_file)
        return

    if args.check_complete:
        done = load_all_done_ids(args.output_file)
        expected = count_jsonl_rows(args.test_file)
        complete = len(done) >= expected
        print(
            f"[check] {args.output_file}: {len(done):,}/{expected:,} "
            f"({'complete' if complete else 'incomplete'})"
        )
        sys.exit(0 if complete else 1)

    if not args.model_path:
        ap.error("--model_path is required unless --check_complete or --merge_only")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world_size, local_rank = _get_rank_world_size()

    if rank == 0:
        print(f"[rank0] prompt_mode={args.prompt_mode}")

    shard_path = f"{args.output_file}.rank{rank}.jsonl"
    done_ids = load_all_done_ids(args.output_file)
    if rank == 0 and done_ids:
        print(
            f"[rank0] Resuming from existing outputs: "
            f"{len(done_ids):,} ids already generated."
        )

    print(
        f"[rank {rank}/{world_size}] local_rank={local_rank} "
        f"output={shard_path}"
    )

    print("=" * 70)
    print(f"[rank {rank}] LOADING MODEL {args.model_path}")
    print("=" * 70)
    load_start = time.time()
    tok_src = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        padding_side="left",
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": f"cuda:{local_rank}"}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[rank {rank}] Model loaded in {time.time() - load_start:.1f}s")

    with open(args.test_file, "r") as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    my_shard = [d for i, d in enumerate(all_data) if i % world_size == rank]
    todo = [d for d in my_shard if d["id"] not in done_ids]
    total_todo = len(todo)
    print(
        f"[rank {rank}] shard={len(my_shard):,} (of {len(all_data):,}) "
        f"todo={total_todo:,}"
    )
    if total_todo == 0:
        print(f"[rank {rank}] Nothing to do.")
        return

    if rank == 0:
        print("[rank0] Warmup ...")
        warm = tokenizer(
            build_inference_prompt(
                {"patient_note": "Test.", "question": "What is the diagnosis?"},
                prompt_mode=args.prompt_mode,
            ),
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            model.generate(
                **warm, max_new_tokens=8, pad_token_id=tokenizer.pad_token_id
            )
        print("[rank0] Warmup done.")

    os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
    out_f = open(shard_path, "a")

    total_tokens = 0
    total_time = 0.0
    done = 0
    run_start = time.time()
    try:
        for start in range(0, total_todo, args.batch_size):
            batch = todo[start : start + args.batch_size]
            results, elapsed = run_batch(model, tokenizer, batch, args)
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total_tokens += r["output_tokens"]
            out_f.flush()
            total_time += elapsed
            done += len(batch)
            if rank == 0 and (
                done % args.log_every < args.batch_size or done == total_todo
            ):
                wall = time.time() - run_start
                tps = total_tokens / total_time if total_time > 0 else 0.0
                eta_s = (
                    (total_todo - done) / (done / wall) if done > 0 else 0.0
                )
                pct = 100.0 * done / total_todo
                print(
                    f"[rank0] {pct:5.1f}% {done:>6,}/{total_todo:,} "
                    f"{tps:.1f} tok/s ETA {eta_s / 3600:.2f}h"
                )
    except KeyboardInterrupt:
        print(f"[rank {rank}] Interrupted — shard saved.")
    finally:
        out_f.close()

    wall_total = time.time() - run_start
    print(
        f"[rank {rank}] DONE shard={done:,} wall={wall_total / 3600:.2f}h "
        f"tok/s={total_tokens / max(total_time, 1e-9):.1f}"
    )


if __name__ == "__main__":
    main()
