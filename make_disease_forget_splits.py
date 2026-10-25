"""Generate disease-level forget/retain splits for an arbitrary disease category.

Same rules/invariants as the Cancer splits:

- Universe is the patients present in ``rest_with_category.jsonl`` (size |U|).
- ``{disease}_forget_X`` contains ``round(X/100 * |U|)`` patients, all of whom
  have ``category == <disease>``.
- ``{disease}_forget_all`` = every patient in the universe with that category.
- Chain: ``{disease}_forget_{p_1} ⊂ ... ⊂ {disease}_forget_{p_k}
         ⊂ {disease}_forget_all``.
- For every target percent ``p``:
    ``{disease}_forget_p  ⊇  {disease-patients}  ∩  forget_p``
  (i.e. all disease-labelled patients that happen to be in the random
  ``forget_p`` split are included in the disease-level forget split).
- ``{disease}_retain_X = rest \\ {disease}_forget_X``.

Each patient has 6 QAs (3 factual + 3 reasoning) in rest, so
``rows == patients * 6`` is asserted per-file.

Existing output files are never overwritten unless ``--force`` is passed.

Usage examples:
    # Cardiovascular: 10, 15, plus "all"
    python scripts/make_disease_forget_splits.py \\
        --disease "Cardiovascular" --percents 10 15 --include-all --seed 0

    # Infectious or Immune Diseases: 10, 15, 20, plus "all"
    python scripts/make_disease_forget_splits.py \\
        --disease "Infectious or Immune Diseases" \\
        --percents 10 15 20 --include-all --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PATIENT_SUFFIX_RE = re.compile(
    r"_(factual|reasoning)"
)
SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def patient_id(record_id: str) -> str:
    return PATIENT_SUFFIX_RE.split(record_id, maxsplit=1)[0]


def slugify(name: str) -> str:
    return SLUG_NON_ALNUM.sub("_", name.strip().lower()).strip("_")


def load_rest(path: Path) -> Tuple[Set[str], Dict[str, int], Dict[str, str]]:
    all_pats: Set[str] = set()
    qa_counts: Dict[str, int] = {}
    cat_of: Dict[str, str] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = patient_id(rec["id"])
            all_pats.add(pid)
            qa_counts[pid] = qa_counts.get(pid, 0) + 1
            cat_of.setdefault(pid, rec.get("category"))
    return all_pats, qa_counts, cat_of


def load_patient_set(path: Path) -> Set[str]:
    patients: Set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            patients.add(patient_id(json.loads(line)["id"]))
    return patients


def stream_filter(
    src: Path, dst: Path, keep: Set[str], *, force: bool
) -> int:
    if dst.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {dst} (pass --force).")
    rows = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            if patient_id(rec["id"]) in keep:
                fout.write(line if line.endswith("\n") else line + "\n")
                rows += 1
    return rows


def build_disease_chain(
    disease_all: Set[str],
    patient_forget: Dict[int, Set[str]],
    percents: List[int],
    target_size,
    rng: random.Random,
) -> Dict[int, Set[str]]:
    """Increasing chain for a single disease.

    For each percent ``p`` (sorted ascending):
      - ``base = prev_forget ∪ (disease_all ∩ patient_forget[p])`` ⊂ disease_all
      - pad from ``disease_all \\ base`` to size ``target_size(p)``.
    """
    chain: Dict[int, Set[str]] = {}
    prev: Set[str] = set()
    for p in percents:
        forced = prev | (disease_all & patient_forget[p])
        if not forced.issubset(disease_all):
            raise AssertionError(
                f"Forced set at p={p} has non-disease patients; should never happen."
            )
        need = target_size(p) - len(forced)
        if need < 0:
            raise ValueError(
                f"Target size for {p}% ({target_size(p)}) is smaller than "
                f"forced set ({len(forced)}). Aborting."
            )
        pool = sorted(disease_all - forced)
        rng.shuffle(pool)
        if need > len(pool):
            raise ValueError(
                f"Not enough disease patients to pad {p}% "
                f"(need {need}, have {len(pool)}). "
                f"The disease pool is too small for this percent."
            )
        cur = forced | set(pool[:need])
        chain[p] = cur
        prev = cur
    return chain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--disease", required=True,
                        help='Exact category string, e.g. "Cardiovascular".')
    parser.add_argument("--slug",
                        help="Filename slug (default: slugified --disease).")
    parser.add_argument("--percents", type=int, nargs="+", required=True,
                        help="Target percents (e.g. 10 15 20).")
    parser.add_argument("--include-all", action="store_true",
                        help=f"Also write {{slug}}_forget_all / {{slug}}_retain_all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite outputs if they exist.")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    rest_path = data_dir / "rest_with_category.jsonl"
    if not rest_path.exists():
        print(f"Missing {rest_path}", file=sys.stderr)
        return 2

    slug = args.slug or slugify(args.disease)
    if not slug:
        print(f"Could not derive a slug from --disease={args.disease!r}",
              file=sys.stderr)
        return 2

    percents = sorted(set(args.percents))
    if not percents:
        print("--percents must be non-empty.", file=sys.stderr)
        return 2

    print(f"Reading universe from {rest_path} ...")
    rest_patients, qa_counts, cat_of = load_rest(rest_path)
    U = len(rest_patients)
    disease_all = {pid for pid, c in cat_of.items() if c == args.disease}
    if not disease_all:
        print(f"No patients in rest have category == {args.disease!r}.",
              file=sys.stderr)
        return 2
    print(f"  |rest| = {U}, |{args.disease}| = {len(disease_all)} (slug: {slug})")

    def target_size(p: int) -> int:
        return round(p / 100 * U)

    max_percent = percents[-1]
    if target_size(max_percent) > len(disease_all):
        print(
            f"Infeasible: target size for {max_percent}% = {target_size(max_percent)} "
            f"exceeds disease pool ({len(disease_all)}).",
            file=sys.stderr,
        )
        return 2

    print(f"\nLoading patient-level forget splits for percents {percents} ...")
    patient_forget: Dict[int, Set[str]] = {}
    for p in percents:
        fp = data_dir / f"forget_{p}_with_category.jsonl"
        if not fp.exists():
            print(f"Missing {fp} (needed to enforce the 'contains random-forget' "
                  f"invariant). Generate it first.", file=sys.stderr)
            return 2
        patient_forget[p] = load_patient_set(fp)
        disease_in_p = len(disease_all & patient_forget[p])
        print(f"  forget_{p}: {len(patient_forget[p])} patients, "
              f"{disease_in_p} with category={args.disease!r}")

    rng = random.Random(args.seed)
    print(f"\nBuilding {slug}_forget chain (seed={args.seed}) ...")
    chain = build_disease_chain(
        disease_all, patient_forget, percents, target_size, rng
    )
    for p in percents:
        print(f"  {slug}_forget_{p}: {len(chain[p])} patients "
              f"(target {target_size(p)})")

    for p1, p2 in zip(percents, percents[1:]):
        if not chain[p1].issubset(chain[p2]):
            raise AssertionError(
                f"{slug}_forget_{p1} is not ⊂ {slug}_forget_{p2}"
            )
    for p in percents:
        assert chain[p].issubset(disease_all)
        forced = disease_all & patient_forget[p]
        assert forced.issubset(chain[p])
        assert len(chain[p]) == target_size(p)

    qa_per_patient = set(qa_counts.values())
    strict_row_check = (qa_per_patient == {6})
    if not strict_row_check:
        print(f"WARNING: QAs/patient observed = {sorted(qa_per_patient)}; "
              f"skipping strict rows == patients*6 check.")

    print(f"\nWriting outputs to {data_dir} ...")
    to_write: List[Tuple[str, Set[str]]] = []
    for p in percents:
        to_write += [
            (f"{slug}_forget_{p}_with_category.jsonl", chain[p]),
            (f"{slug}_retain_{p}_with_category.jsonl", rest_patients - chain[p]),
        ]
    if args.include_all:
        to_write += [
            (f"{slug}_forget_all_with_category.jsonl", disease_all),
            (f"{slug}_retain_all_with_category.jsonl", rest_patients - disease_all),
        ]

    for fname, pset in to_write:
        out = data_dir / fname
        rows = stream_filter(rest_path, out, pset, force=args.force)
        msg = f"  wrote {out.name}: {len(pset)} patients, {rows} rows"
        if strict_row_check:
            assert rows == len(pset) * 6, (
                f"{fname}: rows ({rows}) != patients*6 ({len(pset) * 6})"
            )
            msg += " [patients*6 OK]"
        print(msg)

    print("\nAll invariants satisfied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
