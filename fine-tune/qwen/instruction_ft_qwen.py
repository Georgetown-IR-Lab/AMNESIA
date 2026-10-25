# Adapted From https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
# Modified for Qwen3-8B-Base on AMD MI250X GPUs

#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import io
import json
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset

from utils import *

# NOTE: Flash-attn monkey patch is skipped on AMD MI250X. Qwen3 in recent
# transformers uses SDPA automatically, which works on both ROCm and CUDA.
# Requires transformers >= 4.51.0 (older versions raise KeyError: 'qwen3').
# On NVIDIA you could instead pass attn_implementation="flash_attention_2".

# Stage 1 (train.py) saved only the model weights + config via save_model(), not
# the tokenizer, so MODEL_PATH (=output_pretrain) has no tokenizer files. The
# continued-pretraining step does not change the vocabulary, so the base
# tokenizer is exactly correct. We fall back to it when the checkpoint dir has
# no tokenizer of its own.
BASE_MODEL_FALLBACK = "Qwen/Qwen3-8B-Base"


IGNORE_INDEX = -100

# Kept BYTE-FOR-BYTE identical to the LLaMA-3 run so the prompt format is a
# controlled variable in the comparison. Do not switch this to Qwen ChatML.
PROMPT = """You are an intelligent clinical languge model.
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


def jload(f, mode="r"):
    """Load a .json or .jsonl file into a list of dictionaries."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)

    # Try JSONL first (one JSON object per line), fall back to standard JSON
    content = f.read()
    f.close()

    try:
        # Standard JSON (list of dicts)
        return json.loads(content)
    except json.JSONDecodeError:
        # JSONL format (one dict per line)
        results = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line:
                results.append(json.loads(line))
        return results


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        # Preprocess start/end \n in the data
        for i in range(len(list_data_dict)):
            for k, v in list_data_dict[i].items():
                if isinstance(v, str):
                    list_data_dict[i][k] = v.strip("\n")

        logging.warning("Formatting inputs...")

        # Map field names: your JSONL uses "patient_note", prompt expects "note"
        sources = []
        for example in list_data_dict:
            # Support both "note" and "patient_note" field names
            note = example.get("note", example.get("patient_note", ""))
            sources.append(PROMPT.format(note=note, question=example["question"]))

        # Append EOS to every target so the model learns to stop. For the base
        # model tokenizer.eos_token == "<|endoftext|>" (id 151643), which is the
        # correct stop token used during pretraining.
        targets = [
            f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def load_tokenizer(model_name_or_path, model_max_length):
    """Load the tokenizer from the checkpoint, falling back to the base model.

    Stage-1 continued pre-training saved only the model, so a Stage-1 checkpoint
    dir has no tokenizer files. Since continued pre-training leaves the vocab
    untouched, the base tokenizer is the correct one to use.
    """
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
        )
    except Exception as e:  # noqa: BLE001 -- want any load failure to fall back
        print(
            f"No tokenizer found in {model_name_or_path} ({type(e).__name__}); "
            f"loading tokenizer from {BASE_MODEL_FALLBACK} instead."
        )
        return transformers.AutoTokenizer.from_pretrained(
            BASE_MODEL_FALLBACK,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
        )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("=" * 60)
    print("STAGE 2: INSTRUCTION FINE-TUNING")
    print(f"Model:   {model_args.model_name_or_path}")
    print(f"Data:    {data_args.data_path}")
    print(f"Output:  {training_args.output_dir}")
    if torch.cuda.is_available():
        print(f"GPUs:    {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Incompatible with gradient checkpointing; disable to avoid the warning
    # and the extra memory.
    model.config.use_cache = False

    tokenizer = load_tokenizer(
        model_args.model_name_or_path, training_args.model_max_length
    )
    tokenizer = modify_special_tokens(tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(model=model, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    # Save the tokenizer alongside the model so output_finetune/ is
    # self-contained for downstream evaluation / inference.
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
