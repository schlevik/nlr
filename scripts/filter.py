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

import math
import random
import re
from dataclasses import field, dataclass
from itertools import count

import networkx as nx
import numpy as np
from tqdm import tqdm

from transformers import HfArgumentParser
from typing import Tuple, List

from nlr.common import load_jsonl, from_adj_list, get_all_with_same_num_sents, compare

from scipy.stats import t


@dataclass
class ScriptArguments:
    file: str = field(metadata={"help": "A csv or a json file containing the training data."}
                      )
    out: str
    distinct_to: str = field(
        default=None,
        metadata={"help": "If supplied, will filter out all examples that might be isomorphic to that ds"}
    )
    n_sent: int = field(default=0, metadata={"help": "A csv or a json file containing the training data."})
    proof_len: int = field(default=0, metadata={"help": "Keep only inconsistent problems of this length."})
    equalise: bool = field(default=True, metadata={"help": "Match inconsistent problems with random consistent problems"})
    seed: int = 42

def main():
    parser = HfArgumentParser((ScriptArguments,))

    # noinspection PyTypeChecker
    script_args, *_ = parser.parse_args_into_dataclasses()
    script_args: ScriptArguments
    ds = load_jsonl(script_args.file)
    # check trivial
    if script_args.n_sent:
        out_ds = [d for d in ds if d['sentence'].count('\n') < script_args.n_sent]
    else:
        out_ds = ds
    if script_args.distinct_to:
        other_ds = load_jsonl(script_args.distinct_to)
        compared_ds = compare(ds, other_ds, False)
        out_ds = [d for d in compared_ds if not d['isomorphic_to']]
    if script_args.proof_len:
        out_ds = [d for d in ds if (d.get('proof_len', 0) or 0) > script_args.proof_len]
    if script_args.equalise:
        out_ds.extend(random.sample([d for d in ds if d['label'] == 'consistent'], len(out_ds)))
        random.shuffle(out_ds)
    print(f"Writing dataset of length {len(out_ds)}")
    ds = [json.dumps(p) for p in out_ds]
    with open(script_args.out, 'w+') as f:
        f.write('\n'.join(ds))


if __name__ == "__main__":
    main()
