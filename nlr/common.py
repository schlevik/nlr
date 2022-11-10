import json
from typing import List, Dict, Any

import networkx as nx
import tqdm
from fastcache import clru_cache
from joblib import Parallel, delayed
from networkx.algorithms.isomorphism.isomorph import fast_graph_could_be_isomorphic
from pkg_resources import resource_string


def from_adj_list(adj_list: List[List[int]], directed=True) -> nx.DiGraph:
    g = nx.DiGraph() if directed else nx.Graph()
    for i, l in enumerate(adj_list):
        for j in l:
            g.add_edge(i, j)
    return g

def to_adj_list(g: nx.Graph):
    node_list = []
    for n, ns in g.adjacency():
        edge_list = [n]
        for k in ns.keys():
            edge_list.append(k)
        node_list.append(edge_list)
    return node_list


Data = Dict[str, Any]


def get_all_with_same_num_nodes(dataset: List[Data], n: int):
    return list(filter(lambda d: len(d['graph']) == n, dataset))


def get_all_with_same_num_sents(dataset: List[Data], n: int):
    return list(filter(lambda d: d['sentence'].count('\n') == n, dataset))


def load_jsonl(f, max_l=-1):
    with open(f) as fh:
        lines = fh.readlines()
    return [json.loads(s) for s in (lines if max_l < 0 else lines[:max_l])]


class MyDict(dict):
    def __add__(self, other: dict):
        dct = MyDict()
        for k, v in self.items():
            dct[k] = self[k] + other.get(k, 0)
        return dct

    def __truediv__(self, other):
        dct = MyDict()
        for k, v in self.items():
            dct[k] = self[k] / other
        return dct


def isomorphic(g: List[List[int]], a: List[List[int]]) -> bool:
    g = from_adj_list(g, directed=False)
    a = from_adj_list(a, directed=False)
    return fast_graph_could_be_isomorphic(g, a)


def compare(d1, d2, same_ds):
    @clru_cache(maxsize=len(d2), typed=False)
    def get_stuff(num_nodes, num_sents):
        same_num_sents = get_all_with_same_num_sents(d2, num_sents)
        same_num_nodes = get_all_with_same_num_nodes(same_num_sents, num_nodes)

        return [gg['graph'] for gg in same_num_nodes]

    isomorphics = Parallel(n_jobs=11)(
        delayed(process)(d, get_stuff(len(d['graph']), d['sentence'].count('\n')), same_ds) for d in tqdm.tqdm(d1))
    for d, i in zip(d1, isomorphics):
        d['isomorphic_to'] = i
    return d1


def process(d, all_relevant, same_ds):
    if same_ds:
        return sum(isomorphic(d['graph'], a) for a in all_relevant) > 1
    else:
        return any(isomorphic(d['graph'], a) for a in all_relevant)


_nouns = None


def get_nouns() -> List[str]:
    global _nouns
    _nouns = _nouns or resource_string(__name__, "resources/nouns.txt").decode('utf-8')
    return _nouns.splitlines()


