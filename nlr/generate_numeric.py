import itertools
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count

from typing import List, Tuple, Generator, Dict, Optional

import networkx as nx
from networkx import all_pairs_bellman_ford_path_length
from pkg_resources import resource_string

from nlr.classes import Atom, Literal
from nlr.common import from_adj_list, get_nouns, to_adj_list

logger = logging.getLogger(__name__)


@dataclass(eq=True, unsafe_hash=True)
class NumFormula:
    quantifier: str = field(compare=True, hash=True)
    arg1: Atom = field(compare=True, hash=True)
    # arg1: Literal = field(compare=True, hash=True)
    arg2: Literal = field(compare=True, hash=True)

    def __post_init__(self):
        # if isinstance(self.arg1, Atom):
        #    self.arg1 = Literal(polarity=True, atom=self.arg1)
        assert isinstance(self.arg1, Atom), (self.arg1, type(self.arg1))
        assert isinstance(self.arg2, Literal)

    def to_text(self, realise=False):
        det2 = "an" if (not realise or self.arg2.atom.noun[0] in 'aeio') else 'a'
        arg1 = str(self.arg1) if not realise else self.arg1.noun
        arg2 = str(self.arg2.atom) if not realise else self.arg2.atom.noun
        if self.quantifier == 'all':
            if self.arg2.polarity:
                quantifier = "Every"
            else:
                quantifier = "No"
        elif self.quantifier.startswith('leq'):
            num = self.quantifier.replace('leq', '')
            if self.arg1 == self.arg2.atom:
                return f"There are at most {num} {arg1}s."
            quantifier = f"At most {num}"

        else:
            quantifier = "Some"
        return f"{quantifier} {arg1} is{' not ' if self.quantifier == 'some' and not self.arg2.polarity else ' '}{det2} {arg2}."

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{self.quantifier}({str(self.arg1)},{str(self.arg2)})"

    @property
    def is_universal(self):
        return self.quantifier == 'all'

    @property
    def is_existential(self):
        return self.quantifier == 'some'

    @property
    def atoms(self) -> Tuple[Atom, Atom]:
        return self.arg1, self.arg2.atom


@dataclass
class NumProblem:
    at_most: NumFormula
    universals: List[NumFormula]
    existentials: List[NumFormula]
    atoms: List[Atom]
    depth: int = None
    graph: nx.Graph = None
    realised: bool = False
    _consistent: Optional[bool] = None

    @staticmethod
    def from_formulas(formulas: List[NumFormula], atoms=None):
        atoms = atoms or sorted(list(set(a for f in formulas for a in f.atoms)))
        universals, existentials = [], []
        at_most = None
        for f in formulas:
            if f.is_universal:
                universals.append(f)
            elif f.is_existential:
                existentials.append(f)
            else:
                at_most = f
        assert at_most
        return NumProblem(at_most, universals, existentials, atoms)

    @staticmethod
    def from_str(string: str):
        return NumProblem.from_formulas([NumFormula.from_str(s) for s in string.splitlines()])

    def to_text(self, realise=False, compact=False):
        if compact:
            return self.to_text_compact(realise)
        if realise and not self.realised:
            self.realise()
        return '\n'.join(f.to_text(realise) for f in [self.at_most] + self.universals + self.existentials) + '\n'

    def to_text_compact(self, realise):
        if realise and not self.realised:
            self.realise()

        if realise:
            def get(atom):
                return atom.noun
        else:
            def get(atom):
                return str(atom)

        # atoms appearing in all(p,-q)
        no_ps_are_qs: Dict[str, List[str]] = defaultdict(list)
        for p, q in (f.atoms for f in self.universals):
            no_ps_are_qs[get(p)].append(get(q))
        # atoms appearing in some(p,q)
        some_ps_are_q = [get(f.arg1) for f in self.existentials]
        q = get(self.at_most.arg1)

        at_most = self.at_most.to_text(realise)
        universals = []
        for p, qs in no_ps_are_qs.items():
            det = "an" if (qs[0][0] in 'aeio') else 'a'
            if len(qs) == 1:
                universals.append(f"No {p} is {det} {qs[0]}.")
            else:
                universals.append(
                    f"No {p} is either {' or '.join(f'an {q}' if (q[0] in 'aeio') else f'a {q}' for q in qs)}.")

        det2 = "an" if (at_most[0] in 'aeio') else 'a'
        existentials = f"At least one member of each of the following categories is {det2} {q}: {', '.join(some_ps_are_q)}"

        return '\n'.join([at_most] + universals + [existentials]) + '\n'

    def __str__(self):
        return '\n'.join(str(f) for f in [self.at_most] + self.universals + self.existentials)

    @property
    def consistent(self) -> bool:
        if getattr(self, "_consistent", None) is None:
            raise ValueError("Not classified yet!")
        return self._consistent

    def to_json(self, compact=False):
        result = {
            'label': 'consistent' if self.consistent else 'inconsistent',
            'graph': to_adj_list(self.graph),
            'raw': self.to_text(False),
            'sentence': self.to_text(realise=self.realised, compact=compact)
        }
        return result

    def realise(self, nouns=None):
        nouns = nouns or get_nouns()
        for noun, atom in zip(random.sample(nouns, len(self.atoms)), self.atoms):
            atom.noun = noun
        self.realised = True

    def reorder(self, shuffle=False):
        if shuffle:
            random.shuffle(self.atoms)
        mapping = {y.order: x for x, y in enumerate(self.atoms)}
        for k in self.atoms:
            k.order = mapping[k.order]
        if shuffle:
            random.shuffle(self.universals)
            random.shuffle(self.existentials)


def generate_problem_from_graph(g: nx.Graph, bruteforce=False) -> NumProblem:
    all_atoms, all_sents, pki = preamble(len(g))
    # print(pki)
    for i, j in g.edges:
        all_sents.extend(NumFormula('all', pki[k][i + 1], Literal(False, pki[k][j + 1])) for k in (0, 1, 2))
    p = NumProblem.from_formulas(all_sents, all_atoms)
    p.graph = g
    if bruteforce:
        p._consistent = colour_bruteforce(g)
    else:
        p._consistent = colour_backtracking(g)
    return p


def generate_problem_from_random_graph(num_nodes, transition_prob, nouns=None, seed=None,
                                       do_shuffle=True) -> NumProblem:
    if seed:
        random.seed(seed)

    g = nx.generators.random_graphs.binomial_graph(num_nodes, transition_prob)
    p = generate_problem_from_graph(g, bruteforce=False)
    p.reorder(do_shuffle)

    if nouns is not None:
        p.realise(nouns)

    return p


def generate_random_problem(n):
    all_atoms, all_sents, pki = preamble(n)

    for i, j in ((i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i < j):
        if random.choice([True, False]):
            all_sents.extend(NumFormula('all', pki[k][i], Literal(False, pki[k][j])) for k in (0, 1, 2))
    return NumProblem.from_formulas(all_sents, all_atoms)


def preamble(n):
    all_atoms = []
    p = Atom(0)
    lit_p = Literal(True, p)
    all_atoms.append(p)
    pki = defaultdict(dict)
    ctr = count(1)
    for k in range(0, 3):
        for i in range(1, n + 1):
            a = Atom(next(ctr))
            pki[k][i] = a
            all_atoms.append(a)
    assert len(all_atoms) == 3 * n + 1, len(all_atoms)
    all_sents = [NumFormula("leq3", p, lit_p)]
    all_sents.extend(
        NumFormula('all', pki[j][i], Literal(False, pki[k][i]))
        for i in range(1, n + 1) for j in (0, 1, 2) for k in (0, 1, 2) if j < k
    )
    all_sents.extend(
        NumFormula('some', pki[k][i], Literal(True, p))
        for i in range(1, n + 1) for k in (0, 1, 2)
    )
    assert len(all_sents) == 6 * n + 1, len(all_sents)
    return all_atoms, all_sents, pki


def is_valid_colouring(colouring: Dict[int, int], g: nx.Graph) -> bool:
    for u, v in g.edges:
        if colouring[u] == colouring[v]:
            return False
    return True


def colour_bruteforce(g: nx.Graph, num_colours=3) -> bool:
    return any(is_valid_colouring(dict(enumerate(p)), g) for p in itertools.product(range(num_colours), repeat=len(g)))


def colour_backtracking(g: nx.Graph, num_colours=3) -> bool:
    colours = list(range(num_colours))
    return all(colour_recursive(next(iter(c)), g.subgraph(c), colours) for c in nx.connected_components(g))


def colour_recursive(node, graph: nx.Graph, colours: List[int]):
    for c in colours:
        if any(c == graph.nodes[n].get('colour', None) for n in graph.neighbors(node)):
            # not colourable with this colour
            ...
        else:
            graph.nodes[node]['colour'] = c

            if all(colour_recursive(nbr, graph, colours) for nbr in graph.neighbors(node) if
                   graph.nodes[nbr].get('colour', None) is None):
                return True
    graph.nodes[node]['colour'] = None
    return False
