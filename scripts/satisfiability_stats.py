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
from joblib import delayed, Parallel
from shapely.geometry import LineString
from statsmodels.stats.proportion import proportion_confint

from tqdm import tqdm

from transformers import HfArgumentParser
from typing import Tuple, List

from nlr.common import load_jsonl, from_adj_list, get_all_with_same_num_sents

from scipy.stats import t

from nlr.generate_numeric import colour_backtracking


@dataclass
class ScriptArguments:
    num_runs: int = 1000
    probs_step: int = 5
    graph_sizes: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    njobs: int = 11


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

def get_error_bernoulli(sample, alpha=0.05):
    lower, _ = proportion_confint(sum(sample), len(sample), alpha=alpha)
    return lower


def main():
    parser = HfArgumentParser((ScriptArguments,))

    # noinspection PyTypeChecker
    script_args, *_ = parser.parse_args_into_dataclasses()
    script_args: ScriptArguments
    probs = [x / 100 for x in range(0, 100, script_args.probs_step)]
    intersects = []
    x = range(0, 100, script_args.probs_step)
    fifty_fifty = [0.5] * len(probs)
    plt.plot(x, fifty_fifty)

    for gs in script_args.graph_sizes:
        data = []
        errors = []
        for step in tqdm(probs):
            graphs = [nx.generators.random_graphs.binomial_graph(gs, step) for _ in
                      range(script_args.num_runs)]
            results = Parallel(n_jobs=script_args.njobs)(delayed(colour_backtracking)(g) for g in graphs)
            data.append(sum(results) / script_args.num_runs)
            errors.append(data[-1] - get_error_bernoulli(results))

        plt.title("Path length histogram")
        plt.xticks(range(0, 100, script_args.probs_step))

        plt.errorbar(x=x, y=data, yerr=errors, label=f"{gs} nodes")
        # plt.show()
        data_graph = LineString(np.column_stack((x, data)))
        prob_graph = LineString(np.column_stack((x, fifty_fifty)))
        intersection, _ = data_graph.intersection(prob_graph).xy
        intersects.append(intersection[0]/100)
    print(intersects)
    plt.legend(loc='lower left')
    plt.show()
    plt.plot(script_args.graph_sizes, intersects, label='ratio')
    g = 4.2 / (np.arange(1, max(script_args.graph_sizes)))
    plt.plot(np.arange(1, max(script_args.graph_sizes)), g, label="1/x")
    plt.show()
    plt.legend(loc='upper right')


if __name__ == "__main__":
    main()
