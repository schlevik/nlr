#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch.autograd
from datasets import tqdm

from transformers import (
    AutoTokenizer,
    HfArgumentParser, RobertaTokenizer,
)
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    files: List[str] = field(
        default_factory=list, metadata={"help": "A csv or a json file containing the training data."}
    )
    tokenizer: str = field(default='roberta-base', metadata={"help": "Tokenizer to use."})


def load_jsonl(f):
    with open(f) as fh:
        lines = fh.readlines()
    return [json.loads(s) for s in lines]


def main():
    torch.autograd.set_detect_anomaly(True)
    parser = HfArgumentParser((ScriptArguments,))

    # noinspection PyTypeChecker
    script_args, *_ = parser.parse_args_into_dataclasses()
    script_args: ScriptArguments
    for file in script_args.files:
        d = load_jsonl(file)
        try:
            t = AutoTokenizer.from_pretrained(script_args.tokenizer)
        except:
            t = RobertaTokenizer.from_pretrained(script_args.tokenizer)
        tokens = max(len(t(e['sentence'])['input_ids']) for e in tqdm(d))
        print(f"Max len for file {file}: {tokens}")


if __name__ == "__main__":
    main()
