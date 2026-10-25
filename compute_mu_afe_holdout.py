#!/usr/bin/env python3
"""
Compute MU and AFE on the HOLDOUT set, split by:
  - patient group : forget | retain  (based on which set the patient appears in)
  - QA type       : factual | reasoning | all

Interpretation:
  - holdout-forget group : unseen QAs about forgotten patients
  - holdout-retain group : unseen QAs about retained patients

Metrics follow ``compute_mu_afe_type.py`` definitions (MU, AFE, leakage).
Garbage / refusal follow the same pattern as there:
  mean(garbage_score) and mean(refusal) over QAs where the field is present
  in metrics.generate_qa; counts in n_garbage_checked / n_refusal_checked.
  keyword_leakage_em and garbage_score are written at full float precision.

Outputs:
  mu_afe_holdout_gr.json  -- nested: folder -> group -> qa_type -> {mu, afe, metrics}
  mu_afe_holdout_gr.csv   -- flat: one row per (folder, group, qa_type)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean
from typing import Any, Dict, Optional

from scipy.stats import hmean as scipy_hmean

GROUPS = ["forget", "retain"]
QA_TYPES = ["factual", "reasoning", "all"]
METRICS = ["rougeL", "em", "cosine_similarity", "token_entropy", "entailment_proportion"]

FORGET_FILE = "metrics_by_id_forget.json"
HOLDOUT_FILE = "metrics_by_id_holdout.json"
RETAIN_FILE = "metrics_by_id_retain.json"


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_qa_type(qa_id: str) -> str:
    lower = str(qa_id).lower()
    if "_factual_" in lower:
        return "factual"
    if "_reasoning_" in lower:
        return "reasoning"
    return "unknown"


def safe_hmean(values: list[float]) -> float:
    clamped = [max(1e-9, min(1.0, float(v))) for v in values]
    return round(float(scipy_hmean(clamped)), 4)


def compute_mu(m: dict[str, Any]) -> float:
    return safe_hmean(
        [
            m["rougeL"],
            m["em"],
            m["cosine_similarity"],
            m["token_entropy"],
            m["entailment_proportion"],
        ]
    )


def compute_afe(m: dict[str, Any]) -> float:
    return safe_hmean(
        [
            1.0 - m["rougeL"],
            1.0 - m["em"],
            1.0 - m["cosine_similarity"],
            1.0 - m["entailment_proportion"],
        ]
    )


def empty_bucket() -> dict[str, list]:
    return {
        "rougeL": [],
        "em": [],
        "cosine_similarity": [],
        "token_entropy": [],
        "entailment_labels": [],
        "leakage": [],
        "garbage": [],
        "refusal": [],
    }


def aggregate(bucket: dict[str, list]) -> Optional[dict[str, Any]]:
    n = len(bucket["rougeL"])
    if n == 0:
        return None
    ep = sum(1 for l in bucket["entailment_labels"] if l == "entailment") / n
    m = {
        "rougeL": round(mean(bucket["rougeL"]), 4),
        "em": round(mean(bucket["em"]), 4),
        "cosine_similarity": round(mean(bucket["cosine_similarity"]), 4),
        "token_entropy": round(mean(bucket["token_entropy"]), 4),
        "entailment_proportion": round(ep, 4),
        "keyword_leakage_em": float(mean(bucket["leakage"])) if bucket["leakage"] else None,
        "garbage_score": float(mean(bucket["garbage"])) if bucket["garbage"] else None,
        "refusal": round(mean(bucket["refusal"]), 4) if bucket["refusal"] else None,
        "n": n,
        "n_leakage_checked": len(bucket["leakage"]),
        "n_garbage_checked": len(bucket["garbage"]),
        "n_refusal_checked": len(bucket["refusal"]),
    }
    return {
        "mu": compute_mu(m),
        "afe": compute_afe(m),
        "metrics": m,
    }


def analyze_folder(folder_path: str) -> Optional[dict[str, dict[str, Any]]]:
    paths = {
        "forget": os.path.join(folder_path, FORGET_FILE),
        "holdout": os.path.join(folder_path, HOLDOUT_FILE),
        "retain": os.path.join(folder_path, RETAIN_FILE),
    }
    for key, p in paths.items():
        if not os.path.isfile(p):
            print(f"  [SKIP] missing {key} file: {p}")
            return None

    forget_data = load_json(paths["forget"])
    holdout_data = load_json(paths["holdout"])
    retain_data = load_json(paths["retain"])

    forget_patients = {v["patient_id"] for v in forget_data.values()}
    retain_patients = {v["patient_id"] for v in retain_data.values()}

    buckets: dict[str, dict[str, dict[str, list]]] = {
        g: {qt: empty_bucket() for qt in QA_TYPES} for g in GROUPS
    }
    skipped = 0

    for qa_id, entry in holdout_data.items():
        patient_id = entry["patient_id"]
        m = entry.get("metrics", {}).get("generate_qa", {})

        if patient_id in forget_patients:
            group = "forget"
        elif patient_id in retain_patients:
            group = "retain"
        else:
            skipped += 1
            continue

        qa_type = get_qa_type(str(qa_id))
        if qa_type == "unknown":
            skipped += 1
            continue

        for target_qt in [qa_type, "all"]:
            b = buckets[group][target_qt]
            b["rougeL"].append(float(m.get("rougeL_recall", 0.0)))
            b["em"].append(float(m.get("em", 0.0)))
            b["cosine_similarity"].append(float(m.get("cosine_similarity", 0.0)))
            b["token_entropy"].append(float(m.get("token_entropy", 0.0)))
            b["entailment_labels"].append(str(m.get("entailment_label", "")).lower())
            if m.get("n_keywords_checked", 0) > 0:
                b["leakage"].append(float(m.get("keyword_leakage_em", 0.0)))
            if "garbage_score" in m:
                b["garbage"].append(float(m["garbage_score"]))
            if "refusal" in m:
                b["refusal"].append(float(m["refusal"]))

    if skipped:
        print(f"  Skipped {skipped} holdout QAs (patient not found or unknown qa type)")

    results: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        results[group] = {}
        for qt in QA_TYPES:
            results[group][qt] = aggregate(buckets[group][qt])

    return results


def print_folder(folder_name: str, r: dict[str, dict[str, Any]]) -> None:
    print(
        f"\n  {'Group':<10} {'QA type':<12} {'MU':>7} {'AFE':>7} "
        f"{'Leakage':>9} {'Garbage':>9} {'Refusal':>9}  {'n':>7}"
    )
    print("  " + "-" * 75)
    for group in GROUPS:
        for qt in QA_TYPES:
            v = r[group][qt]
            if v is None:
                print(f"  {group:<10} {qt:<12}    N/A     N/A       N/A       N/A       N/A")
                continue
            met = v["metrics"]
            leak = met["keyword_leakage_em"]
            leak_str = f"{leak:.4f}" if leak is not None else "N/A"
            garb = met["garbage_score"]
            garb_str = f"{garb:.4f}" if garb is not None else "N/A"
            ref = met["refusal"]
            ref_str = f"{ref:.4f}" if ref is not None else "N/A"
            print(
                f"  {group:<10} {qt:<12} {v['mu']:>7.4f} {v['afe']:>7.4f} "
                f"{leak_str:>9} {garb_str:>9} {ref_str:>9}  {met['n']:>7,}"
            )

    print(f"\n  {'Group':<10} {'QA type':<12} " + "  ".join(f"{m[:12]:>14}" for m in METRICS))
    print("  " + "-" * (22 + 16 * len(METRICS)))
    for group in GROUPS:
        for qt in QA_TYPES:
            v = r[group][qt]
            if v is None:
                continue
            vals = "  ".join(f"{v['metrics'][m]:>14.4f}" for m in METRICS)
            print(f"  {group:<10} {qt:<12} {vals}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        default="/saves/eval_recomputed_new",
        help="Directory containing experiment subfolders.",
    )
    ap.add_argument("--output-json", default="mu_afe_holdout_gr.json")
    ap.add_argument("--output-csv", default="mu_afe_holdout_gr.csv")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    folders = sorted(
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and not d.startswith(".")
        and "_all_" not in d
    )
    if not folders:
        print(f"No subfolders found in {base_dir}")
        return 2

    print(f"Found {len(folders)} folder(s) in '{base_dir}'")

    all_results: Dict[str, dict] = {}
    for folder_name in folders:
        print(f"\n{'=' * 60}\n  {folder_name}")
        r = analyze_folder(os.path.join(base_dir, folder_name))
        if r is None:
            continue
        all_results[folder_name] = r
        print_folder(folder_name, r)

    if not all_results:
        print("\nNo results.")
        return 1

    print(f"\n\n{'=' * 90}")
    print("  SUMMARY  (holdout set, split by forget/retain patient membership)")
    print(f"{'=' * 90}")
    cols = [(g, qt) for g in GROUPS for qt in ["factual", "reasoning", "all"]]
    hdr = f"  {'Folder':<42}" + "".join(
        f" {g[:3] + '/' + qt[:3] + ' MU':>11} {'AFE':>7}" for g, qt in cols
    )
    print(hdr)
    print("  " + "-" * len(hdr))
    for folder_name, r in all_results.items():
        row = f"  {folder_name:<42}"
        for g, qt in cols:
            v = r[g][qt]
            if v is None:
                row += f"{'N/A':>12} {'N/A':>7}"
            else:
                row += f" {v['mu']:>11.4f} {v['afe']:>7.4f}"
        print(row)

    json_out = os.path.join(base_dir, args.output_json)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] {json_out}")

    csv_out = os.path.join(base_dir, args.output_csv)
    fieldnames = (
        ["folder", "group", "qa_type", "mu", "afe"]
        + list(METRICS)
        + [
            "keyword_leakage_em",
            "n_leakage_checked",
            "garbage_score",
            "n_garbage_checked",
            "refusal",
            "n_refusal_checked",
            "n",
        ]
    )
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for folder_name, r in all_results.items():
            for group in GROUPS:
                for qt in QA_TYPES:
                    v = r[group][qt]
                    if v is None:
                        continue
                    row: dict[str, Any] = {
                        "folder": folder_name,
                        "group": group,
                        "qa_type": qt,
                        "mu": v["mu"],
                        "afe": v["afe"],
                    }
                    for m_name in METRICS:
                        row[m_name] = v["metrics"][m_name]
                    row["keyword_leakage_em"] = v["metrics"]["keyword_leakage_em"]
                    row["n_leakage_checked"] = v["metrics"]["n_leakage_checked"]
                    row["garbage_score"] = v["metrics"]["garbage_score"]
                    row["n_garbage_checked"] = v["metrics"]["n_garbage_checked"]
                    row["refusal"] = v["metrics"]["refusal"]
                    row["n_refusal_checked"] = v["metrics"]["n_refusal_checked"]
                    row["n"] = v["metrics"]["n"]
                    writer.writerow(row)
    print(f"[Saved] {csv_out}")
    print(f"\nDone. Processed {len(all_results)} folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
