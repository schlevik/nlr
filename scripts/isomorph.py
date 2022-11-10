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
from dataclasses import field, dataclass

from transformers import HfArgumentParser
from typing import List

from nlr.common import get_all_with_same_num_sents, load_jsonl, compare


@dataclass
class ScriptArguments:
    files: List[str] = field(
        default_factory=list, metadata={"help": "A csv or a json file containing the training data."}
    )
    same_ds: bool = field(
        default=False
    )


def main():
    parser = HfArgumentParser((ScriptArguments,))

    # noinspection PyTypeChecker
    script_args, *_ = parser.parse_args_into_dataclasses()
    script_args: ScriptArguments
    d1, d2 = [load_jsonl(file) for file in script_args.files]

    compare(d1, d2, same_ds=script_args.same_ds)

    def key(d):
        return d['sentence'].count('\n')

    overall = 0
    min_sents = key(min(d1, key=key))
    max_sents = key(max(d1, key=key)) + 1
    for i in range(min_sents, max_sents):
        v_is = get_all_with_same_num_sents(d1, i)
        if v_is:
            results = sum(v_i['isomorphic_to'] for v_i in v_is)
            results = results / len(v_is)
            print(f"For n={i}: {results:.3f}")
            overall += results
    print("overall: ")
    print(overall / (max_sents - min_sents))


if __name__ == "__main__":
    main()
