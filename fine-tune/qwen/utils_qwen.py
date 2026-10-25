# Adapted from Asclepius utils.py for Qwen3-8B-Base

from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import Trainer


def modify_special_tokens(tokenizer):
    """Configure special tokens for Qwen3-8B-Base.

    Qwen3 uses the Qwen2Tokenizer (byte-level BPE). Unlike LLaMA:
      * It has NO bos_token (bos_token == None, add_bos_token == False).
        Do not rely on tokenizer.bos_token_id -- it is None.
      * For the BASE model, both eos_token and pad_token are "<|endoftext|>"
        (id 151643). This is the end-of-text token that was actually used
        during pretraining, so it is also the correct document separator for
        continued pre-training. (The post-trained/instruct Qwen3-8B instead
        uses eos_token "<|im_end|>" (151645) for ChatML -- not relevant here.)

    Because pad_token is already defined, the `if pad_token is None` guard
    below does nothing for a stock Qwen3-Base tokenizer; it is kept only as a
    defensive fallback. Note that for the base model pad_token == eos_token,
    which is harmless for this pipeline since packed fixed-length sequences
    never require padding.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-8B-Base")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": (
                "Maximum sequence length. Sequences will be right padded (and "
                "possibly truncated). Qwen3-8B supports up to 32768 natively; "
                "2048 is kept as a memory-friendly default for continued "
                "pre-training on patient notes."
            )
        },
    )
    remove_unused_columns: bool = field(
        default=False,
    )
    dataloader_num_workers: int = field(
        default=7,
    )


# ─── Prompt templates ───────────────────────────────────────────────
# Same clinical prompts as original Asclepius, kept for compatibility.
# NOTE: These are only used for downstream instruction-tuning / inference,
# not for the Stage-1 continued pre-training in this repo.

PROMPT_DICT = {
    "ours": """You are an intelligent clinical languge model.
Below is a snippet of patient's discharge summary and a following instruction from healthcare professional.
Write a response that appropriately completes the instruction.
The response should provide the accurate answer to the instruction, while being concise.

[Discharge Summary Begin]
{note}
[Discharge Summary End]

[Instruction Begin]
{question}
[Instruction End] 
""",
    "alpaca": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Input:\n{note}\n\n### Response:"
    ),
    "medalpaca": (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
        "\n\n### Instruction:\n{question}\n\n### Input:\n{note}\n\n### Response:\n"
    ),
    "chat": """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: [The start of the Discharge Summary]
{note}
[The end of the Discharge Summary]
{question} ASSISTANT: 
""",
    # Qwen3 native ChatML template (thinking mode left to the caller via
    # tokenizer.apply_chat_template if preferred). Provided as a convenience
    # for downstream instruction-tuning that wants to match Qwen's format.
    "qwen": (
        "<|im_start|>system\n"
        "You are an intelligent clinical language model. Given a snippet of a "
        "patient's discharge summary and an instruction, write a concise, "
        "accurate response.<|im_end|>\n"
        "<|im_start|>user\n"
        "[Discharge Summary Begin]\n{note}\n[Discharge Summary End]\n\n"
        "[Instruction Begin]\n{question}\n[Instruction End]<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}


def get_prompt(model_name):
    name = model_name.lower()
    if "qwen" in name:
        print("Using Qwen ChatML Prompt")
        return PROMPT_DICT["qwen"]
    elif "llama-3" in name or "meta-llama" in name:
        print("Using Our Prompt (LLaMA 3)")
        return PROMPT_DICT["ours"]
    elif model_name in ["decapoda-research/llama-7b-hf", "chaoyi-wu/PMC_LLAMA_7B"]:
        print("Using Ours+Response Prompt")
        return PROMPT_DICT["ours"] + "\nResponse: "
    elif model_name in [
        "chavinlo/alpaca-native",
        "zl111/ChatDoctor",
    ]:
        print("Using Alpaca Prompt")
        return PROMPT_DICT["alpaca"]
    elif model_name == "medalpaca/medalpaca-7b":
        print("Using MedAlpaca Prompt")
        return PROMPT_DICT["medalpaca"]
    elif "vicuna" in name or "clinical-camel" in name:
        print("Using Vicuna Prompt")
        return PROMPT_DICT["chat"]
    else:
        print("Using Our Prompt")
        return PROMPT_DICT["ours"]
