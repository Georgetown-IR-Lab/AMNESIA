# Adapted From: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
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


from dataclasses import dataclass
from typing import Dict

import torch
import transformers
from datasets import load_from_disk

from utils import *

# NOTE: The original Asclepius code uses a flash_attn monkey patch for A100 GPUs.
# On AMD MI250X, we skip this because:
#   1. The flash_attn package is built for NVIDIA CUDA, not AMD ROCm
#   2. Qwen3 in recent transformers already supports efficient attention
#      via SDPA (torch.nn.functional.scaled_dot_product_attention) which
#      works on both NVIDIA and AMD GPUs automatically.
#
# Requirements: transformers >= 4.51.0 (older versions raise
#   KeyError: 'qwen3' because the Qwen3 architecture is unknown to them).
#
# If running on A100s, you can uncomment the lines below (note: the Asclepius
# flash-attn monkey patch is LLaMA-specific and does NOT apply to Qwen3; on
# NVIDIA you would instead pass attn_implementation="flash_attention_2" to
# from_pretrained after installing flash-attn):
#
# if "A100" in torch.cuda.get_device_name():
#     model_kwargs["attn_implementation"] = "flash_attention_2"


@dataclass
class Collator(object):
    def __call__(self, instances):
        input_ids = torch.stack([torch.LongTensor(i["input_ids"]) for i in instances])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def make_data_module(data_args) -> Dict:
    train_dataset = load_from_disk(data_args.data_path)
    data_collator = Collator()
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("=" * 60)
    print("STAGE 1: CONTINUED PRE-TRAINING")
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
        # SDPA works on both ROCm and CUDA; explicit for clarity.
        attn_implementation="sdpa",
    )
    # use_cache is incompatible with gradient checkpointing during training;
    # disable it to silence the warning and avoid the extra memory.
    model.config.use_cache = False

    # Load the tokenizer and pass it to the Trainer so save_model() writes the
    # tokenizer files (special_tokens_map.json, tokenizer.json,
    # tokenizer_config.json) alongside the weights -- matching the LLaMA
    # pipeline's output layout so Stage 2 can load everything from the
    # checkpoint dir directly.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        use_fast=True,
    )
    tokenizer = modify_special_tokens(tokenizer)

    data_module = make_data_module(data_args=data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        **data_module,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    # Explicit save as well, so the tokenizer is present regardless of the
    # transformers version's save_model() behavior.
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
