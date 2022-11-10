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
import math
import re
from dataclasses import field, dataclass
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from tqdm import tqdm

from transformers import HfArgumentParser
from typing import Tuple, List

from nlr.common import load_jsonl, from_adj_list, get_all_with_same_num_sents

from scipy.stats import t


@dataclass
class ScriptArguments:
    file: str = field(metadata={"help": "A csv or a json file containing the training data."}
                      )


pattern_some = re.compile(r'Some (.+) is not an \1\.')


def find_some_p_are_not_p(problem: str) -> List[Tuple[str, str]]:
    return re.findall(pattern_some, problem)


#
#
# def find_if_no_p_are_q(p, q, problem: str) -> bool:
#     return f"No {p} is an {q}." in problem
#
#
# pattern_all = re.compile(r'Every (.)+ is an (.)+.')
#
#
# def find_all_p_are_q(problem: str):
#     return re.findall(pattern_all, problem)
#
#
# def find_if_some_p_is_not_q(p, q, problem: str) -> bool:
#     return f"Some {p} is not an {q}." in problem

def get_mean_var_ci(sample, alpha=0.025):
    sample = np.array(sample)
    t_ci = t.ppf(1 - alpha, df=len(sample) - 1)
    return sample.mean(), sample.var(), t_ci * sample.std() / math.sqrt(len(sample))


def main():
    parser = HfArgumentParser((ScriptArguments,))

    # noinspection PyTypeChecker
    script_args, *_ = parser.parse_args_into_dataclasses()
    script_args: ScriptArguments
    ds = load_jsonl(script_args.file)
    # check trivial
    num_trivial = sum(len(find_some_p_are_not_p(p["sentence"])) > 0 for p in ds)
    print(f'num trivial problems: {num_trivial}/{len(ds)} = {num_trivial / len(ds):.3f}')
    print(
        f"Just identifying trivial and then guessing consistent: {(num_trivial + sum(d['label'] == 'consistent' for d in ds)) / len(ds):.3f}")
    depths = [d['depth'] for d in ds if d['label'] == 'inconsistent']
    mean, var, ci = get_mean_var_ci(depths)
    plt.title("Path length histogram")
    print(f'average depth: {mean:.3f} (+/-{ci}). Max depth: {max(depths)}')
    plt.xticks(range(0, max(depths)))
    plt.hist(depths, bins=max(depths), align='left')
    plt.show()
    plt.clf()
    incons_per_problems = [len(d['all_depths']) for d in ds if d['label'] == 'inconsistent']
    mean, var, ci = get_mean_var_ci(incons_per_problems)
    print(f'average inconsistent per inconsistent example: {mean:.3f} (+/-{ci}), Max: {max(incons_per_problems)}')
    plt.title("Inconsistencies per example histogram")
    plt.xticks(range(0, max(incons_per_problems)))
    plt.hist(incons_per_problems, bins=max(incons_per_problems), align='left')
    plt.show()

    def key(d):
        return d['sentence'].count('\n')

    min_sents = key(min(ds, key=key))
    max_sents = key(max(ds, key=key)) + 1
    for i in range(min_sents, max_sents):
        all_incons_with_num_sents = [d for d in get_all_with_same_num_sents(ds, i) if d['label'] == 'inconsistent']
        incons_per_problems = [len(d['all_depths']) for d in all_incons_with_num_sents]
        depths = np.array([d['depth'] for d in all_incons_with_num_sents])

        mean, var, ci = get_mean_var_ci(depths)
        print(f'average depth for {i}: {mean:.3f} (+/- {ci:.3f}). Depth to length ratio: {mean / i:.3f}')


if __name__ == "__main__":
    main()
