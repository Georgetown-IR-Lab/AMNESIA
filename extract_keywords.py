#!/usr/bin/env python3
"""
Fast disease-related keyword extraction from patient notes, questions, and answers.

Input JSONL rows should contain:
  - patient_note
  - question and/or answer fields when available
  - disease_name OR disease OR category OR diagnosis OR condition
  - optional id

Output JSONL rows are written in real time, one row per processed patient.

The script processes only the main three disease categories:
  - Cancer
  - Cardiovascular
  - Infectious or Immune Diseases

For each row, the compact output contains only:
  - id, question, answer, patient_note, category
  - note_keywords and note_keywords_with_scores: top K disease-related keywords from patient_note
  - question_keywords and question_keywords_with_scores: all relevant disease-related keywords from question
  - answer_keywords and answer_keywords_with_scores: all relevant disease-related keywords from answer
  - all_keywords and all_keywords_with_scores: merged unique keywords from note, question, and answer

Recommended install:
  pip install spacy scispacy tqdm
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz

Patient notes are always processed with en_core_sci_lg (word-vector model, no token
limit) so that the full note is parsed regardless of length.  Questions and answers
are processed with --qa_model (defaults to en_core_sci_lg as well; set to
en_core_sci_scibert if you want transformer-quality embeddings on the shorter QA texts).

Example:
  python extract_keywords_two_files_for_unlearning.py \
    --inputs hold_out_with_category.jsonl rest_with_category.jsonl \
    --output_dir data/keyword_outputs \
    --note_top_k 5 \
    --question_top_k 0 \
    --answer_top_k 0

  # Use SciBERT for questions/answers, lg for notes:
  python extract_keywords_two_files_for_unlearning.py \
    --inputs hold_out_with_category.jsonl \
    --qa_model en_core_sci_scibert
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import spacy
from tqdm import tqdm


# Notes are always processed with the lg (word-vector) model so that the full note
# text is parsed without the 512-token truncation that affects SciBERT.
# Questions and answers are shorter and can optionally use a different model.
NOTE_MODEL = "en_core_sci_lg"
DEFAULT_QA_MODEL = "en_core_sci_lg"

TARGET_CATEGORIES = {
    "Cancer",
    "Cardiovascular",
    "Infectious or Immune Diseases",
}

# Disease/category seed terms used for lexical disease relevance scoring.
DISEASE_SEEDS: Dict[str, List[str]] = {
    "cancer": [
        # General oncology
        "cancer", "tumor", "tumour", "neoplasm", "malignancy", "malignant",
        "carcinoma", "sarcoma", "lymphoma", "leukemia", "leukaemia",
        "myeloma", "glioma", "blastoma",
        # Pathology
        "metastasis", "metastatic", "adenocarcinoma", "melanoma",
        "squamous cell", "hepatocellular", "biopsy", "cytology",
        "grade", "staging", "margins", "excision",
        # Specific cancers, preferably as compound terms
        "lung cancer", "breast cancer", "colon cancer", "prostate cancer",
        "ovarian cancer", "pancreatic cancer", "lymph node",
        "bone marrow", "recurrence", "relapse",
        # Treatment
        "chemotherapy", "radiotherapy", "radiation therapy",
        "adjuvant", "neoadjuvant", "palliative",
        "targeted therapy", "hormone therapy", "resection", "debulking",
        "pet scan", "screening", "remission", "immunotherapy",
    ],

    "cardiovascular": [
        # Core terms
        "cardiovascular", "cardiac", "coronary", "myocardial", "infarction",
        "angina", "arrhythmia", "atrial fibrillation", "ventricular",
        "heart failure", "cardiomyopathy", "pericarditis",
        # Conditions
        "hypertension", "atherosclerosis", "ischemia", "ischaemia",
        "stroke", "aortic", "mitral", "valve", "embolism",
        "thrombosis", "dvt", "pulmonary embolism",
        "bradycardia", "tachycardia", "syncope", "heart attack",
        # Biomarkers
        "troponin", "bnp", "ntprobnp", "ejection fraction",
        "lipids", "cholesterol", "statin",
        # Procedures and abbreviations
        "stent", "cabg", "bypass", "angioplasty", "revascularization",
        "echocardiogram", "ecg", "ekg", "defibrillation", "pacemaker",
        "catheterization", "chest pain", "cad", "chf", "hfref", "hfpef",
    ],

    "infectious or immune diseases": [
        # Infection general
        "infection", "infectious", "sepsis", "bacteremia", "fever",
        "abscess", "cellulitis", "wound infection",
        # Pathogens
        "viral", "virus", "bacterial", "bacteria", "fungal", "candida",
        "pathogen", "mrsa", "tuberculosis", "covid",
        "hiv", "hepatitis", "influenza",
        # Specific infections
        "pneumonia", "meningitis", "encephalitis", "osteomyelitis", "uti",
        # Immune system
        "immune", "immunity", "autoimmune", "immunosuppression",
        "immunocompromised", "complement", "cytokine", "interferon",
        "rheumatoid", "lupus", "vasculitis", "seropositive",
        # Inflammation markers
        "inflammation", "inflammatory", "crp", "esr",
        "leukocytosis", "neutrophil", "lymphocyte", "wbc",
        # Diagnostics and treatment
        "antibiotic", "antiviral", "antifungal", "prophylaxis",
        "culture", "pcr", "serology",
    ],
}

# Terms that are common in clinical notes but usually bad keywords.
BAD_TERMS = {
    "patient", "patients", "year", "years", "old", "male", "female", "history",
    "medical history", "past medical history", "pmh", "today", "yesterday", "day",
    "days", "week", "weeks", "month", "months", "time", "normal", "none",
    "no", "yes", "denies", "reported", "reports", "present", "mg", "ml", "cm",
    "right", "left", "bilateral", "mild", "moderate", "severe", "acute", "chronic",
    "follow", "follow up", "plan", "assessment", "exam", "physical exam",
    "question", "answer", "following", "option", "options", "true", "false",
}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+/-]*")

DISEASE_FIELDS = ("disease_name", "disease", "category", "diagnosis", "condition")
NOTE_FIELDS = ("patient_note", "note", "clinical_note", "text")
QUESTION_FIELDS = ("question", "Question", "query", "prompt")
ANSWER_FIELDS = ("answer", "Answer", "response", "label", "target", "completion")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;()[]{}'\"")
    return text


def get_first_text(row: dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_disease_name(row: dict) -> str:
    """Accept several likely column names without forcing the user to rename input."""
    return get_first_text(row, DISEASE_FIELDS)


def get_patient_note(row: dict) -> str:
    return get_first_text(row, NOTE_FIELDS)


def get_question_text(row: dict) -> str:
    return get_first_text(row, QUESTION_FIELDS)


def get_answer_text(row: dict) -> str:
    return get_first_text(row, ANSWER_FIELDS)


def seed_terms_for_disease(disease_name: str) -> List[str]:
    """Return seeds for broad known categories plus tokens from the disease name itself."""
    disease_norm = normalize_text(disease_name)
    seeds: List[str] = []

    # Include disease name and its informative words.
    if disease_norm:
        seeds.append(disease_norm)
        seeds.extend(w.group(0).lower() for w in WORD_RE.finditer(disease_norm))

    # FIX (Bug 4): Track which category seed sets have already been added to avoid
    # double-adding them when both the loop and the alias block match the same category.
    added_categories: set = set()

    # Add broad category seeds if the disease/category matches or contains a known domain.
    for category, category_seeds in DISEASE_SEEDS.items():
        if category in disease_norm or disease_norm in category:
            seeds.extend(category_seeds)
            added_categories.add(category)

    # Handle common aliases — only add if not already included by the loop above.
    if any(x in disease_norm for x in ("heart", "cardiac", "cardio", "vascular")):
        if "cardiovascular" not in added_categories:
            seeds.extend(DISEASE_SEEDS["cardiovascular"])
            added_categories.add("cardiovascular")
    if any(x in disease_norm for x in ("cancer", "tumor", "tumour", "neoplasm", "oncology", "malignan")):
        if "cancer" not in added_categories:
            seeds.extend(DISEASE_SEEDS["cancer"])
            added_categories.add("cancer")
    if any(x in disease_norm for x in ("infect", "immune", "viral", "bacterial", "autoimmune")):
        if "infectious or immune diseases" not in added_categories:
            seeds.extend(DISEASE_SEEDS["infectious or immune diseases"])
            added_categories.add("infectious or immune diseases")

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for s in seeds:
        s = normalize_text(s)
        if s and s not in seen and s not in BAD_TERMS:
            seen.add(s)
            out.append(s)
    return out


def is_good_phrase(phrase: str) -> bool:
    p = normalize_text(phrase)
    if not p or p in BAD_TERMS:
        return False
    if len(p) < 3:
        return False
    if len(p.split()) > 6:
        return False
    if not WORD_RE.search(p):
        return False
    if p.isdigit():
        return False
    return True


def word_tuple(text: str) -> Tuple[str, ...]:
    return tuple(w.group(0).lower() for w in WORD_RE.finditer(normalize_text(text)))


def contains_word_sequence(longer: Tuple[str, ...], shorter: Tuple[str, ...]) -> bool:
    """Return True when shorter appears as a contiguous word sequence in longer."""
    if not shorter or len(shorter) > len(longer):
        return False
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def lexical_related_score(phrase: str, seeds: List[str]) -> float:
    """
    Fast lexical relevance score.
    This avoids expensive pairwise embedding similarity while still filtering by disease.

    Compound multi-word seeds (e.g. "lung cancer", "heart failure") are
    now matched as whole phrases rather than being fragmented into individual words.
    A single-word phrase like "cancer" no longer scores 0.9 against "lung cancer"
    via substring containment; instead only exact-match or whole-phrase containment
    is rewarded at the high-confidence level.  Word-overlap scoring now compares
    the phrase against each *word* of the seed independently so that partial
    single-word matches still get partial credit, but multi-word seeds are not
    conflated with any of their constituent words.
    """
    phrase_norm = normalize_text(phrase)
    phrase_words = word_tuple(phrase_norm)
    phrase_word_set = set(phrase_words)
    if not phrase_word_set:
        return 0.0

    score = 0.0
    for seed in seeds:
        seed_norm = normalize_text(seed)
        seed_words = word_tuple(seed_norm)
        seed_word_set = set(seed_words)
        if not seed_word_set:
            continue

        if phrase_norm == seed_norm:
            # Exact match
            score = max(score, 1.0)
        elif len(seed_words) > 1:
            # Multi-word seed: only reward if the phrase contains the whole seed
            # phrase as a contiguous sequence (or vice-versa).
            if contains_word_sequence(phrase_words, seed_words):
                score = max(score, 0.9)
            elif contains_word_sequence(seed_words, phrase_words):
                # Phrase is a sub-sequence of the seed (e.g. phrase="lung", seed="lung cancer")
                # Give a modest partial credit rather than the full 0.9.
                score = max(score, 0.5)
            else:
                overlap = len(phrase_word_set & seed_word_set) / max(
                    1, min(len(phrase_word_set), len(seed_word_set))
                )
                score = max(score, overlap * 0.6)
        else:
            # Single-word seed: substring or word-level match
            if phrase_norm in seed_norm or seed_norm in phrase_norm:
                score = max(score, 0.9)
            else:
                overlap = len(phrase_word_set & seed_word_set) / max(
                    1, min(len(phrase_word_set), len(seed_word_set))
                )
                score = max(score, overlap)

    return score


def extract_candidates(doc) -> List[str]:
    """Extract medical entities and noun chunks from a parsed scispaCy/spaCy doc."""
    candidates: List[str] = []

    # scispaCy models produce broad biomedical entities here.
    for ent in doc.ents:
        text = normalize_text(ent.text)
        if is_good_phrase(text):
            candidates.append(text)

    # Noun chunks help when the model misses useful clinical phrases.
    for chunk in doc.noun_chunks:
        text = normalize_text(chunk.text)
        if is_good_phrase(text):
            candidates.append(text)

    return candidates


def remove_redundant_longer_phrases(
    scored: List[Tuple[str, float, int, float]],
    min_subphrase_words: int = 1,
) -> List[Tuple[str, float, int, float]]:
    """
    Remove longer phrases when they contain a shorter selected candidate.

    Examples:
      keep    "lung cancer"
      remove  "small tumor lung cancer"

      keep    "aortic"
      remove  "aortic cannula"

    This uses strict shortest-phrase preference. If a shorter keyword appears
    as a contiguous word sequence inside a longer keyword from the same sample,
    the longer keyword is removed before top-k selection. The next highest
    ranked non-redundant candidate then fills the top-k list when available.
    """
    if not scored:
        return scored

    # Prefer shortest phrasing first, then higher score/frequency.
    # These kept phrases become blockers for longer redundant phrases.
    by_shortness = sorted(
        scored,
        key=lambda x: (len(word_tuple(x[0])), len(x[0]), -x[1], -x[2]),
    )

    kept_blockers: List[Tuple[str, Tuple[str, ...]]] = []
    removed = set()

    for phrase, score, freq, rel in by_shortness:
        p_words = word_tuple(phrase)
        if not p_words:
            continue

        redundant = False
        for kept_phrase, kept_words in kept_blockers:
            if len(kept_words) < min_subphrase_words:
                continue
            if len(kept_words) < len(p_words) and contains_word_sequence(p_words, kept_words):
                redundant = True
                break

        if redundant:
            removed.add(phrase)
        else:
            kept_blockers.append((phrase, p_words))

    return [item for item in scored if item[0] not in removed]


def select_top_keywords(
    scored: List[Tuple[str, float, int, float]],
    top_k: int,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Sort candidates and select top_k.
    If top_k <= 0, return all ranked candidates.
    """
    scored.sort(key=lambda x: (x[1], x[2], len(x[0])), reverse=True)
    scored = remove_redundant_longer_phrases(scored)
    scored.sort(key=lambda x: (x[1], x[2], len(x[0])), reverse=True)

    selected = scored if top_k <= 0 else scored[:top_k]
    keywords = [phrase for phrase, _, _, _ in selected]
    scores = {phrase: score for phrase, score, _, _ in selected}
    return keywords, scores


def rank_keywords(
    candidates: List[str],
    seeds: List[str],
    top_k: int,
    min_score: float,
) -> Tuple[List[str], Dict[str, float]]:
    counts = Counter(candidates)
    scored: List[Tuple[str, float, int, float]] = []

    for phrase, freq in counts.items():
        rel = lexical_related_score(phrase, seeds)
        if rel < min_score:
            continue

        # Frequency helps repeated important terms rise, but does not dominate relevance.
        score = rel + min(freq, 5) * 0.05
        scored.append((phrase, round(score, 4), freq, rel))

    return select_top_keywords(scored, top_k=top_k)


def fallback_keywords(candidates: List[str], disease_name: str, top_k: int) -> Tuple[List[str], Dict[str, float]]:
    """Fallback if strict disease filtering finds nothing."""
    counts = Counter(candidates)
    scored: List[Tuple[str, float, int, float]] = []
    disease_words = set(word_tuple(disease_name))
    # Remove generic label tokens so fallback does not reward uninformative words.
    disease_words.discard("diseases")

    for phrase, freq in counts.items():
        words = set(word_tuple(phrase))
        if not words:
            continue
        # Prefer phrases sharing terms with disease name, then frequency, then longer medical phrases.
        overlap = len(words & disease_words)
        score = overlap * 0.5 + min(freq, 5) * 0.05 + min(len(words), 4) * 0.02
        scored.append((phrase, round(score, 4), freq, float(overlap)))

    return select_top_keywords(scored, top_k=top_k)


def merge_keyword_outputs(
    first_keywords: List[str],
    first_scores: Dict[str, float],
    second_keywords: List[str],
    second_scores: Dict[str, float],
) -> Tuple[List[str], Dict[str, float]]:
    """Merge keyword lists while preserving order and keeping the maximum score."""
    merged: List[str] = []
    merged_scores: Dict[str, float] = {}
    for phrase in first_keywords + second_keywords:
        if phrase not in merged_scores:
            merged.append(phrase)
            merged_scores[phrase] = max(first_scores.get(phrase, 0.0), second_scores.get(phrase, 0.0))
        else:
            merged_scores[phrase] = max(merged_scores[phrase], first_scores.get(phrase, 0.0), second_scores.get(phrase, 0.0))
    return merged, merged_scores


def extract_for_doc(
    doc,
    disease_name: str,
    seeds: List[str],
    top_k: int,
    min_score: float,
    use_fallback: bool,
) -> Tuple[List[str], Dict[str, float], Dict[str, int], str]:
    """Extract and rank disease-related keywords for one parsed text."""
    candidates = extract_candidates(doc)
    keywords, scores = rank_keywords(candidates, seeds, top_k=top_k, min_score=min_score)
    method = "scispacy_entities_disease_filtered"

    if not keywords and use_fallback:
        keywords, scores = fallback_keywords(candidates, disease_name, top_k=top_k)
        method = "scispacy_entities_fallback"

    return keywords, scores, dict(Counter(candidates).most_common(50)), method


def load_rows(path: Path) -> Iterable[Tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_no}: {e}", file=sys.stderr)
                continue
            yield line_no, row


def load_model(model_name: str):
    try:
        # en_core_sci_lg has no transformer or tagger; disable only what exists.
        # spaCy silently ignores names that aren't present in the pipeline.
        return spacy.load(model_name, disable=["tagger", "lemmatizer"])
    except OSError as e:
        raise SystemExit(
            f"Could not load spaCy/scispaCy model '{model_name}'.\n\n"
            f"Install it first, for example:\n"
            f"  pip install scispacy\n"
            f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model_name}-0.5.4.tar.gz\n\n"
            f"Original error: {e}"
        )


def dedupe_keywords_with_scores_preserve_order(
    keyword_score_pairs: Sequence[Tuple[List[str], Dict[str, float]]],
) -> Tuple[List[str], Dict[str, float]]:
    """
    Merge keyword lists with scores, remove exact duplicates, and remove longer
    redundant phrases when they contain a shorter selected keyword.

    Order is preserved by source priority: note, then question, then answer.
    If the same keyword appears in multiple sources, the maximum score is kept.
    """
    ordered: List[str] = []
    scores: Dict[str, float] = {}

    for keywords, keyword_scores in keyword_score_pairs:
        for phrase in keywords:
            p = normalize_text(phrase)
            if not p:
                continue
            score = float(keyword_scores.get(phrase, keyword_scores.get(p, 0.0)))
            if p not in scores:
                ordered.append(p)
                scores[p] = score
            else:
                scores[p] = max(scores[p], score)

    # Identify longer redundant phrases using the same rule as note/question/answer selection,
    # but return the remaining phrases in the original note -> question -> answer order.
    scored_for_redundancy = [(p, scores[p], 1, scores[p]) for p in ordered]
    kept_after_redundancy = remove_redundant_longer_phrases(scored_for_redundancy)
    kept_set = {p for p, _, _, _ in kept_after_redundancy}

    final_keywords = [p for p in ordered if p in kept_set]
    final_scores = {p: round(scores[p], 4) for p in final_keywords}
    return final_keywords, final_scores


def extract_keyword_list_for_doc(
    doc,
    disease_name: str,
    seeds: List[str],
    top_k: int,
    min_score: float,
    use_fallback: bool,
) -> Tuple[List[str], Dict[str, float], str]:
    """Return the keyword list, scores, and method label."""
    keywords, scores, candidates, method = extract_for_doc(
        doc,
        disease_name=disease_name,
        seeds=seeds,
        top_k=top_k,
        min_score=min_score,
        use_fallback=use_fallback,
    )
    return keywords, scores, method


def output_path_for_input(input_file: Path, output_dir: Path) -> Path:
    """Create a stable output name for each input split."""
    stem = input_file.stem
    return output_dir / f"{stem}_keywords.jsonl"


def process_file(
    input_path: str,
    output_path: str,
    nlp_note,
    nlp_qa,
    note_top_k: int = 5,
    question_top_k: int = 0,
    answer_top_k: int = 0,
    min_score: float = 0.25,
    batch_size: int = 32,
    n_process: int = 1,
    fsync: bool = False,
):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise SystemExit(f"Input file not found: {input_file}")

    all_rows = list(load_rows(input_file))
    rows = [(line_no, row) for line_no, row in all_rows if get_disease_name(row) in TARGET_CATEGORIES]
    excluded_rows = len(all_rows) - len(rows)

    print(f"\nInput file         : {input_file}")
    print(f"Output file        : {output_file}")
    print(f"Loaded rows        : {len(all_rows)}")
    print(f"Target rows        : {len(rows)}")
    print(f"Excluded rows      : {excluded_rows}")
    print(f"Target categories  : {sorted(TARGET_CATEGORIES)}")
    # FIX (Bug 6 / API clarity): Explain the top_k=0 sentinel in the run summary.
    print(f"Note top_k         : {note_top_k} ({'all' if note_top_k <= 0 else note_top_k})")
    print(f"Question top_k     : {question_top_k} ({'all' if question_top_k <= 0 else question_top_k})")
    print(f"Answer top_k       : {answer_top_k} ({'all' if answer_top_k <= 0 else answer_top_k})")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    note_texts = [get_patient_note(row) for _, row in rows]
    question_texts = [get_question_text(row) for _, row in rows]
    answer_texts = [get_answer_text(row) for _, row in rows]

    # Notes use the lg model (no token-length cap); Q&A uses the qa model.
    note_docs = nlp_note.pipe(note_texts, batch_size=batch_size, n_process=n_process)
    question_docs = nlp_qa.pipe(question_texts, batch_size=batch_size, n_process=n_process)
    answer_docs = nlp_qa.pipe(answer_texts, batch_size=batch_size, n_process=n_process)

    processed = 0
    empty_notes = 0
    empty_questions = 0
    empty_answers = 0
    note_fallback_rows = 0

    with output_file.open("w", encoding="utf-8") as out:
        iterator = zip(rows, note_texts, question_texts, answer_texts, note_docs, question_docs, answer_docs)
        for (line_no, row), patient_note, question, answer, note_doc, question_doc, answer_doc in tqdm(
            iterator, total=len(rows), desc=f"Extracting {input_file.name}"
        ):
            patient_id = row.get("id", f"line_{line_no}")
            category = get_disease_name(row)
            seeds = seed_terms_for_disease(category)

            if patient_note.strip():
                note_keywords, note_scores, note_method = extract_keyword_list_for_doc(
                    note_doc,
                    disease_name=category,
                    seeds=seeds,
                    top_k=note_top_k,
                    min_score=min_score,
                    use_fallback=True,
                )
                if note_method == "scispacy_entities_fallback":
                    note_fallback_rows += 1
            else:
                note_keywords = []
                note_scores = {}
                empty_notes += 1

            if question.strip():
                question_keywords, question_scores, _ = extract_keyword_list_for_doc(
                    question_doc,
                    disease_name=category,
                    seeds=seeds,
                    top_k=question_top_k,
                    min_score=min_score,
                    use_fallback=False,
                )
            else:
                question_keywords = []
                question_scores = {}
                empty_questions += 1

            if answer.strip():
                answer_keywords, answer_scores, _ = extract_keyword_list_for_doc(
                    answer_doc,
                    disease_name=category,
                    seeds=seeds,
                    top_k=answer_top_k,
                    min_score=min_score,
                    use_fallback=False,
                )
            else:
                answer_keywords = []
                answer_scores = {}
                empty_answers += 1

            all_keywords, all_scores = dedupe_keywords_with_scores_preserve_order(
                [
                    (note_keywords, note_scores),
                    (question_keywords, question_scores),
                    (answer_keywords, answer_scores),
                ]
            )

            # Requested compact output format.
            result = {
                "id": patient_id,
                "question": question,
                "answer": answer,
                "patient_note": patient_note,
                "category": category,
                "note_keywords": note_keywords,
                "note_keywords_with_scores": note_scores,
                "question_keywords": question_keywords,
                "question_keywords_with_scores": question_scores,
                "answer_keywords": answer_keywords,
                "answer_keywords_with_scores": answer_scores,
                "all_keywords": all_keywords,
                "all_keywords_with_scores": all_scores,
            }

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if fsync:
                os.fsync(out.fileno())
            processed += 1

    print("\n=== SUMMARY ===")
    print(f"Processed rows        : {processed}")
    print(f"Excluded rows         : {excluded_rows}")
    print(f"Empty patient notes   : {empty_notes}")
    print(f"Empty questions       : {empty_questions}")
    print(f"Empty answers         : {empty_answers}")
    print(f"Note fallback rows    : {note_fallback_rows}")
    print(f"Saved to              : {output_file}")


def process(
    inputs: List[str],
    output_dir: str = "data/keyword_outputs",
    qa_model: str = DEFAULT_QA_MODEL,
    note_top_k: int = 5,
    question_top_k: int = 0,
    answer_top_k: int = 0,
    min_score: float = 0.25,
    batch_size: int = 32,
    n_process: int = 1,
    fsync: bool = False,
):
    # Notes always use en_core_sci_lg (word-vector, no 512-token cap).
    print(f"Loading note model : {NOTE_MODEL}")
    nlp_note = load_model(NOTE_MODEL)
    # Guard against scispaCy pickle errors with multiprocessing.
    if n_process > 1:
        try:
            import pickle
            pickle.dumps(nlp_note)
        except Exception:
            print(
                f"WARNING: note model '{NOTE_MODEL}' is not picklable — "
                "forcing n_process=1 to avoid multiprocessing errors.",
                file=sys.stderr,
            )
            n_process = 1
    # Q&A model may be the same or a different (e.g. transformer) model.
    if qa_model == NOTE_MODEL:
        print(f"Loading QA model   : {qa_model} (same as note model, reusing)")
        nlp_qa = nlp_note
    else:
        print(f"Loading QA model   : {qa_model}")
        nlp_qa = load_model(qa_model)
        if n_process > 1:
            try:
                import pickle
                pickle.dumps(nlp_qa)
            except Exception:
                print(
                    f"WARNING: QA model '{qa_model}' is not picklable — "
                    "forcing n_process=1 to avoid multiprocessing errors.",
                    file=sys.stderr,
                )
                n_process = 1

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for input_path in inputs:
        input_file = Path(input_path)
        out_path = output_path_for_input(input_file, output_root)
        process_file(
            input_path=str(input_file),
            output_path=str(out_path),
            nlp_note=nlp_note,
            nlp_qa=nlp_qa,
            note_top_k=note_top_k,
            question_top_k=question_top_k,
            answer_top_k=answer_top_k,
            min_score=min_score,
            batch_size=batch_size,
            n_process=n_process,
            fsync=fsync,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract disease-related keywords from hold-out/rest JSONL files.\n"
            "Patient notes are always processed with en_core_sci_lg (no token-length cap).\n"
            "Questions and answers use --qa_model (default: en_core_sci_lg)."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["hold_out_with_category.jsonl", "rest_with_category.jsonl"],
        help="Input JSONL files to process.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/keyword_outputs",
        help="Directory where one output JSONL will be written per input file.",
    )
    parser.add_argument(
        "--qa_model",
        default=DEFAULT_QA_MODEL,
        help=(
            "scispaCy model for questions and answers "
            f"(default: {DEFAULT_QA_MODEL}). "
            "Set to en_core_sci_scibert for transformer-quality embeddings on Q&A."
        ),
    )
    parser.add_argument("--note_top_k", type=int, default=5, help="Number of keywords to return from patient_note.")
    parser.add_argument(
        "--question_top_k",
        type=int,
        default=0,
        help="Max keywords from question. Use 0 (default) to return all relevant keywords. Use a negative value to disable question keyword extraction entirely.",
    )
    parser.add_argument(
        "--answer_top_k",
        type=int,
        default=0,
        help="Max keywords from answer. Use 0 (default) to return all relevant keywords. Use a negative value to disable answer keyword extraction entirely.",
    )
    parser.add_argument("--min_score", type=float, default=0.5, help="Disease relevance threshold. Lower = more keywords.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Pipe batch size. en_core_sci_lg can handle larger batches than SciBERT; 32 is a good default.",
    )
    parser.add_argument(
        "--n_process",
        type=int,
        default=8,
        help="Number of parallel processes for nlp.pipe. Safe to increase for en_core_sci_lg; keep at 1 for SciBERT QA model.",
    )
    parser.add_argument("--fsync", action="store_true", help="Flush to disk after every row. Safer but slower.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        inputs=args.inputs,
        output_dir=args.output_dir,
        qa_model=args.qa_model,
        note_top_k=args.note_top_k,
        question_top_k=args.question_top_k,
        answer_top_k=args.answer_top_k,
        min_score=args.min_score,
        batch_size=args.batch_size,
        n_process=args.n_process,
        fsync=args.fsync,
    )