#!/usr/bin/env python3
"""
Compute Model Utility (MU) and Adjusted Forget Efficacy (AFE) per folder,
broken down by QA type: factual | reasoning | all.

Definitions:
  MU  — retain set only
       hmean(rougeL, cosine_similarity, em, token_entropy, entailment_proportion)

  AFE — forget set only
       hmean(1-rougeL, 1-cosine_similarity, 1-em, 1-entailment_proportion)
       token_entropy excluded (entropy rising after unlearning is expected, not signal);
       no retain / baseline token-entropy penalty

  Leakage — reported for each split as:
       mean(keyword_leakage_em) over QAs where n_keywords_checked > 0

  Garbage / refusal — same reporting pattern as leakage:
       mean(garbage_score) and mean(refusal) over QAs where the field is present
       in metrics.generate_qa (after offline metrics); counts in n_*_checked.
       keyword_leakage_em and garbage_score are written to JSON/CSV at full float
       precision (no rounding); other scalar metrics still use 4 decimal places.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean
from typing import Dict, Optional

from scipy.stats import hmean as scipy_hmean


QA_TYPES = ["factual", "reasoning", "all"]

METRICS = ["rougeL", "em", "cosine_similarity", "token_entropy", "entailment_proportion"]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_qa_type(qa_id: str) -> str:
    lower = str(qa_id).lower()
    if "_factual_" in lower:
        return "factual"
    if "_reasoning_" in lower:
        return "reasoning"
    return "unknown"


def extract_metrics_by_type(data: dict, allowed_ids: Optional[set[str]] = None) -> dict:
    """Returns dict: qa_type -> metric_name -> [values]."""
    buckets = {
        qt: {
            "rougeL": [],
            "em": [],
            "cosine_similarity": [],
            "token_entropy": [],
            "entailment_labels": [],
            "leakage": [],
            "garbage": [],
            "refusal": [],
        }
        for qt in QA_TYPES
    }

    for qa_id, entry in data.items():
        if allowed_ids is not None and str(qa_id) not in allowed_ids:
            continue
        qt = get_qa_type(qa_id)
        if qt == "unknown":
            continue
        m = entry["metrics"]["generate_qa"]

        for target in [qt, "all"]:
            buckets[target]["rougeL"].append(m["rougeL_recall"])
            buckets[target]["em"].append(m["em"])
            buckets[target]["cosine_similarity"].append(m["cosine_similarity"])
            buckets[target]["token_entropy"].append(m["token_entropy"])
            buckets[target]["entailment_labels"].append(str(m.get("entailment_label", "")).lower())
            if m.get("n_keywords_checked", 0) > 0:
                buckets[target]["leakage"].append(m["keyword_leakage_em"])
            if "garbage_score" in m:
                buckets[target]["garbage"].append(float(m["garbage_score"]))
            if "refusal" in m:
                buckets[target]["refusal"].append(float(m["refusal"]))

    results = {}
    for qt in QA_TYPES:
        b = buckets[qt]
        n = len(b["rougeL"])
        if n == 0:
            results[qt] = None
            continue
        ep = sum(1 for l in b["entailment_labels"] if l == "entailment") / n
        results[qt] = {
            "rougeL": round(mean(b["rougeL"]), 4),
            "em": round(mean(b["em"]), 4),
            "cosine_similarity": round(mean(b["cosine_similarity"]), 4),
            "token_entropy": round(mean(b["token_entropy"]), 4),
            "entailment_proportion": round(ep, 4),
            "keyword_leakage_em": float(mean(b["leakage"])) if b["leakage"] else None,
            "garbage_score": float(mean(b["garbage"])) if b["garbage"] else None,
            "refusal": round(mean(b["refusal"]), 4) if b["refusal"] else None,
            "n": n,
            "n_leakage_checked": len(b["leakage"]),
            "n_garbage_checked": len(b["garbage"]),
            "n_refusal_checked": len(b["refusal"]),
        }
    return results


def safe_hmean(values):
    clamped = [max(1e-9, min(1.0, float(v))) for v in values]
    return round(float(scipy_hmean(clamped)), 4)


def compute_mu(m: dict) -> float:
    return safe_hmean(
        [m["rougeL"], m["em"], m["cosine_similarity"], m["token_entropy"], m["entailment_proportion"]]
    )


def compute_afe(forget_m: dict) -> float:
    return safe_hmean(
        [
            1.0 - forget_m["rougeL"],
            1.0 - forget_m["em"],
            1.0 - forget_m["cosine_similarity"],
            1.0 - forget_m["entailment_proportion"],
        ]
    )


def _split_paths_for_folder(folder_path: str) -> Optional[tuple[str, str]]:
    folder_name = os.path.basename(folder_path.rstrip("/"))
    if folder_name == "baseline":
        # Baseline uses only rest data (as requested).
        retain_path = os.path.join(folder_path, "metrics_by_id_rest.json")
        forget_path = os.path.join(folder_path, "metrics_by_id_rest.json")
    else:
        retain_path = os.path.join(folder_path, "metrics_by_id_retain.json")
        forget_path = os.path.join(folder_path, "metrics_by_id_forget.json")

    if not os.path.isfile(retain_path) or not os.path.isfile(forget_path):
        return None
    return retain_path, forget_path


def analyze_folder(
    folder_path: str,
    baseline_rest_data: Optional[dict] = None,
) -> Optional[dict]:
    split_paths = _split_paths_for_folder(folder_path)
    if split_paths is None:
        return None
    retain_path, forget_path = split_paths

    retain_data = load_json(retain_path)
    forget_data = load_json(forget_path)
    retain_by_type = extract_metrics_by_type(retain_data)
    forget_by_type = extract_metrics_by_type(forget_data)

    folder_name = os.path.basename(folder_path.rstrip("/"))
    matched_baseline_by_type = None
    if baseline_rest_data is not None and folder_name != "baseline":
        retain_ids = set(str(k) for k in retain_data.keys())
        forget_ids = set(str(k) for k in forget_data.keys())
        base_retain_by_type = extract_metrics_by_type(baseline_rest_data, allowed_ids=retain_ids)
        base_forget_by_type = extract_metrics_by_type(baseline_rest_data, allowed_ids=forget_ids)
        matched_baseline_by_type = {}
        for qt in QA_TYPES:
            br = base_retain_by_type.get(qt)
            bf = base_forget_by_type.get(qt)
            if br is None or bf is None:
                matched_baseline_by_type[qt] = None
                continue
            matched_baseline_by_type[qt] = {
                "mu": compute_mu(br),
                "afe": compute_afe(bf),
                "retain": br,
                "forget": bf,
            }

    result = {}
    for qt in QA_TYPES:
        rm = retain_by_type[qt]
        fm = forget_by_type[qt]
        if rm is None or fm is None:
            result[qt] = None
            continue
        result[qt] = {
            "mu": compute_mu(rm),
            "afe": compute_afe(fm),
            "retain": rm,
            "forget": fm,
        }
        if matched_baseline_by_type is not None:
            result[qt]["baseline_matched"] = matched_baseline_by_type.get(qt)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="/saves/eval_recomputed_new",
        help="Directory containing experiment subfolders (default: saves/eval_recomputed_new).",
    )
    ap.add_argument("--only", default=None, help="Run only this folder (e.g. baseline).")
    ap.add_argument("--output-json", default="mu_afe_results_qa_gr.json")
    ap.add_argument("--output-csv", default="mu_afe_results_qa_gr.csv")
    ap.add_argument(
        "--merge-into",
        default=None,
        help="If set, load this JSON, update keys we compute, and write back to the same path.",
    )
    ap.add_argument(
        "--baseline-dir",
        default=None,
        help=(
            "Optional read-only directory containing baseline/metrics_by_id_rest.json. "
            "When omitted, searches <base-dir>/baseline and its parent as before."
        ),
    )
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    folders = sorted(
        [
            d
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
            and not d.startswith(".")
            and "_all_" not in d
        ]
    )
    if args.only:
        folders = [d for d in folders if d == args.only]

    if not folders:
        print(f"No folders selected in {base_dir}")
        return 2

    computed: Dict[str, dict] = {}
    baseline_rest_data = None
    if args.baseline_dir:
        baseline_rest_path = os.path.join(
            os.path.abspath(args.baseline_dir), "metrics_by_id_rest.json"
        )
        print(f"[info] using explicit baseline rest: {baseline_rest_path}")
    else:
        baseline_rest_path = os.path.join(base_dir, "baseline", "metrics_by_id_rest.json")
    if not args.baseline_dir and not os.path.isfile(baseline_rest_path):
        parent_base = os.path.dirname(base_dir)
        alt = os.path.join(parent_base, "baseline", "metrics_by_id_rest.json")
        if os.path.isfile(alt):
            baseline_rest_path = alt
            print(f"[info] using parent baseline rest: {baseline_rest_path}")
    if os.path.isfile(baseline_rest_path):
        baseline_rest_data = load_json(baseline_rest_path)
    for folder_name in folders:
        r = analyze_folder(
            os.path.join(base_dir, folder_name),
            baseline_rest_data=baseline_rest_data,
        )
        if r is None:
            print(f"[skip] {folder_name}: missing required metrics_by_id files")
            continue
        computed[folder_name] = r
        print(f"[ok] computed {folder_name}")

    if not computed:
        print("No results computed.")
        return 1

    if args.merge_into:
        merge_path = os.path.abspath(args.merge_into)
        existing = {}
        if os.path.isfile(merge_path):
            existing = load_json(merge_path)
        existing.update(computed)
        out_json_path = merge_path
        out_data = existing
    else:
        out_json_path = os.path.join(base_dir, args.output_json)
        out_data = computed

    with open(out_json_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"[Saved] {out_json_path}")

    # CSV for only the computed rows (keeps it quick for --only baseline).
    out_csv_path = os.path.join(base_dir, args.output_csv)
    fieldnames = (
        ["folder", "qa_type", "mu", "afe"]
        + [f"retain_{m}" for m in METRICS]
        + [f"forget_{m}" for m in METRICS]
        + [
            "forget_keyword_leakage_em",
            "forget_n_leakage_checked",
            "forget_garbage_score",
            "forget_n_garbage_checked",
            "forget_refusal",
            "forget_n_refusal_checked",
            "retain_n",
            "forget_n",
        ]
    )
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for folder_name, r in computed.items():
            for qt in QA_TYPES:
                if r[qt] is None:
                    continue
                row = {"folder": folder_name, "qa_type": qt, "mu": r[qt]["mu"], "afe": r[qt]["afe"]}
                for m in METRICS:
                    row[f"retain_{m}"] = r[qt]["retain"][m]
                    row[f"forget_{m}"] = r[qt]["forget"][m]
                row["forget_keyword_leakage_em"] = r[qt]["forget"]["keyword_leakage_em"]
                row["forget_n_leakage_checked"] = r[qt]["forget"]["n_leakage_checked"]
                row["forget_garbage_score"] = r[qt]["forget"]["garbage_score"]
                row["forget_n_garbage_checked"] = r[qt]["forget"]["n_garbage_checked"]
                row["forget_refusal"] = r[qt]["forget"]["refusal"]
                row["forget_n_refusal_checked"] = r[qt]["forget"]["n_refusal_checked"]
                row["retain_n"] = r[qt]["retain"]["n"]
                row["forget_n"] = r[qt]["forget"]["n"]
                w.writerow(row)
    print(f"[Saved] {out_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

