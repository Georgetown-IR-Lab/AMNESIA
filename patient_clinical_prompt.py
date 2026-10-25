"""Clinical QA prompt shared by fine-tuning and evaluation.
"""

from typing import Mapping, Union

# Identical to instruction_ft.PROMPT / run_inference.PROMPT
FINETUNE_CLINICAL_PROMPT = """You are an intelligent clinical languge model.
Below is a snippet of patient's discharge summary and a following instruction from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

[Discharge Summary Begin]
{note}
[Discharge Summary End]

[Instruction Begin]
{question}
[Instruction End]
"""

NOTE_FIELDS = ("patient_note", "note", "clinical_note", "text")


def patient_note_from_row(row: Mapping[str, object]) -> str:
    for key in NOTE_FIELDS:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def bare_question_from_row(row: Mapping[str, object]) -> str:
    """Return the short clinical question, not an already-wrapped prompt."""
    q = row.get("question")
    if not isinstance(q, str):
        return ""
    q = q.strip()
    if not q:
        return ""
    if q.startswith("You are an intelligent clinical languge model."):
        return ""
    if q.startswith("You are a clinical QA assistant."):
        return ""
    return q


def build_finetune_prompt(row: Union[Mapping[str, object], dict]) -> str:
    """Full note-conditioned prompt used during fine-tuning."""
    note = patient_note_from_row(row)
    question = bare_question_from_row(row)
    if not question:
        qraw = row.get("question")
        if isinstance(qraw, str) and qraw.strip():
            question = qraw.strip()
    if not question:
        raise ValueError("Missing question in row id={!r}".format(row.get("id")))
    return FINETUNE_CLINICAL_PROMPT.format(note=note, question=question)


def build_bare_qa_prompt(row: Union[Mapping[str, object], dict]) -> str:
    """Closed-book eval: clinical question only, no discharge summary."""
    question = bare_question_from_row(row)
    if not question:
        qraw = row.get("question")
        if isinstance(qraw, str) and qraw.strip():
            question = qraw.strip()
    if not question:
        raise ValueError("Missing question in row id={!r}".format(row.get("id")))
    return question


def build_inference_prompt(
    row: Union[Mapping[str, object], dict], *, prompt_mode: str = "finetune"
) -> str:
    if prompt_mode in ("finetune", "note", "with_note"):
        return build_finetune_prompt(row)
    if prompt_mode in ("bare_qa", "no_note", "closed_book", "qa"):
        return build_bare_qa_prompt(row)
    raise ValueError(f"Unknown prompt_mode={prompt_mode!r}")
