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
from enum import Enum

import math
import random
import re
from dataclasses import field, dataclass
from transformers import HfArgumentParser
from itertools import count

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from typing import Tuple, List, Generator, Callable

from nlr.common import load_jsonl, from_adj_list, get_all_with_same_num_sents

from scipy.stats import t

from nlr.generate import get_hard_problem, get_guaranteed_depth, get_balanced_with_guaranteed_depth
from nlr.generate_numeric import generate_problem_from_random_graph


class GenerationStrategy(Enum):
    Graph = 'graph'

    @classmethod
    def from_str(cls, string):
        return next(v for n, v in cls.__members__.items() if v.value == string or v == string)


@dataclass
class ScriptArguments:
    out: str
    strategy: GenerationStrategy = field(default=GenerationStrategy.Graph, metadata={'help': "ee"})
    min_num_nodes: int = 5
    max_num_nodes: int = 20
    k: int = 500
    factor: float = 4.2
    seed: int = 42
    param_generator: Callable[['ScriptArguments'], Generator] = field(init=False)
    gen_method: Callable = field(init=False)
    shuffle: bool = True
    compact: int = True
    realise: str = None

    def __post_init__(self):
        self.strategy = GenerationStrategy.from_str(self.strategy)
        if self.strategy == GenerationStrategy.Graph:
            self.param_generator = param_generator_graph
            self.gen_method = generate_problem_from_random_graph
        else:
            raise NotImplementedError()


def param_generator_graph(script_args: ScriptArguments) -> Generator[Tuple[int, int, int, int], None, None]:
    total_len = (script_args.max_num_nodes - script_args.min_num_nodes + 1) * script_args.k
    print(total_len)
    seeds = iter(random.sample(range(1000 * total_len), total_len))
    if script_args.realise is not None:
        with open(script_args.realise) as f:
            nouns = [n for n in f.read().splitlines() if n.strip()]
    else:
        nouns = None
    for nodes in range(script_args.min_num_nodes, script_args.max_num_nodes + 1):
        p = script_args.factor / nodes
        for _ in range(script_args.k):
            yield nodes, p, nouns, next(seeds), script_args.shuffle


def main():
    parser = HfArgumentParser((ScriptArguments,))

    # noinspection PyTypeChecker
    script_args, *_ = parser.parse_args_into_dataclasses()
    script_args: ScriptArguments
    random.seed(script_args.seed)
    gen = script_args.param_generator(script_args)

    problems = Parallel(n_jobs=11)(delayed(script_args.gen_method)(*args) for args in tqdm(list(gen)))

    print(f"{sum(p.consistent for p in problems)} consistent.")
    print(f"{sum(not p.consistent for p in problems)} inconsistent.")
    ds = [json.dumps(p.to_json(script_args.compact)) for p in problems]
    output = '\n'.join(ds) + '\n'

    if script_args.strategy == GenerationStrategy.Graph:
        descriptor = "numeric-graph"
    else:
        raise NotImplementedError()
    if not script_args.shuffle:
        descriptor += '-unshuffled'
    if script_args.realise:
        descriptor += '-realised'
    if script_args.compact:
        descriptor += '-compact'
    out_file = "-".join([
        script_args.out,
        descriptor,
        f"s{script_args.seed}",
        f"mi{script_args.min_num_nodes}",
        f"ma{script_args.max_num_nodes}",
        f"k{script_args.k}"
    ]) + '.json'
    print(f"Saving under {out_file}.")
    if script_args.out:
        with open(out_file, 'w+') as f:
            f.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
