import logging
import random
from dataclasses import dataclass, field
from itertools import count

from typing import List, Tuple, Generator

import networkx as nx
from networkx import all_pairs_bellman_ford_path_length

from nlr.classes import Atom, Literal
from nlr.common import from_adj_list, get_nouns, to_adj_list

logger = logging.getLogger(__name__)


@dataclass(eq=True, unsafe_hash=True)
class Formula:
    quantifier: str = field(compare=True, hash=True)
    # arg1: Atom = field(compare=True, hash=True)
    arg1: Literal = field(compare=True, hash=True)
    arg2: Literal = field(compare=True, hash=True)

    def __post_init__(self):
        if isinstance(self.arg1, Atom):
            self.arg1 = Literal(polarity=True, atom=self.arg1)
        assert isinstance(self.arg1, Literal), (self.arg1, type(self.arg1))
        assert isinstance(self.arg2, Literal)

    @classmethod
    def from_str(cls, string) -> 'Formula':
        string = string.lower()
        quantifier = 'some' if string.startswith('some') else 'all'
        tokens = string.split()
        arg1 = Atom(int(tokens[1][1:]), var=tokens[1][0])
        arg2_int = int(tokens[-1][1:-1])
        arg2 = Literal(string.startswith('every') or (string.startswith('some') and len(tokens) <= 5),
                       Atom(arg2_int, var=tokens[-1][0]))
        return Formula(quantifier, arg1, arg2)

    def to_text(self, realise=False):
        if self.quantifier == 'all':
            if self.arg2.polarity:
                quantifier = "Every"
            else:
                quantifier = "No"
        else:
            quantifier = "Some"
        det2 = "an" if (not realise or self.arg2.atom.noun[0] in 'aeio') else 'a'
        arg1 = str(self.arg1.atom) if not realise else self.arg1.atom.noun
        arg2 = str(self.arg2.atom) if not realise else self.arg2.atom.noun
        return f"{quantifier} {'non-' if not self.arg1.polarity else ''}{arg1} is{' not ' if self.quantifier == 'some' and not self.arg2.polarity else ' '}{det2} {arg2}."

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
        return self.arg1.atom, self.arg2.atom


@dataclass
class SatProblem:
    universals: List[Formula]
    existentials: List[Formula]
    atoms: List[Atom]
    depth: int = None
    graph: List[List[int]] = None
    all_depths: List[int] = field(default_factory=list)
    realised: bool = False

    @staticmethod
    def from_formulas(formulas: List[Formula], atoms=None):
        atoms = atoms or sorted(list(set(a for f in formulas for a in f.atoms)))
        universals, existentials = [], []
        for f in formulas:
            if f.is_universal:
                universals.append(f)
            else:
                existentials.append(f)
        return SatProblem(universals, existentials, atoms)

    @staticmethod
    def from_str(string: str):
        return SatProblem.from_formulas([Formula.from_str(s) for s in string.splitlines()])

    def to_text(self, realise=False):
        return '\n'.join(f.to_text(realise) for f in self.universals + self.existentials) + '\n'

    def __str__(self):
        return '\n'.join(str(f) for f in self.universals + self.existentials)

    @property
    def consistent(self):
        return self.depth == 0

    def to_json(self):
        result = {
            "depth": self.depth,
            'all_depths': self.all_depths,
            'label': 'consistent' if self.consistent else 'inconsistent',
            'graph': self.graph,
            'raw': self.to_text(False),
            'sentence': self.to_text(realise=self.realised)
        }
        return result

    def realise(self, nouns=None):
        nouns = nouns or get_nouns()
        for noun, atom in zip(random.sample(nouns, len(self.atoms)), self.atoms):
            atom.noun = noun
        self.realised = True

    def reorder(self, shuffle=False):
        # used_atoms = list(a for f in self.universals + self.existentials for a in f.atoms)
        # print(len(used_atoms))
        # print([str(a) for a in used_atoms])
        # print(set(str(a) for a in used_atoms))
        if shuffle:
            random.shuffle(self.atoms)
        # print([(y.order, x) for x, y in enumerate(self.atoms)])
        mapping = {y.order: x for x, y in enumerate(self.atoms)}
        # print({y: x for y, x in mapping.items()})
        for k in self.atoms:
            k.order = mapping[k.order]
        # print([str(a) for a in used_atoms])
        if shuffle:
            random.shuffle(self.universals)
            random.shuffle(self.existentials)


class Indices:
    def __init__(self, atoms):
        self.indices = {a: i for i, a in enumerate(atoms)}
        self.indices_reversed = {i: a for i, a in enumerate(atoms)}

    def __getitem__(self, item: Formula):
        u = 2 * self.indices[item.arg1.atom]
        v = 2 * self.indices[item.arg2.atom]
        if not item.arg1.polarity:
            u += 1
        if not item.arg2.polarity:
            v += 1
        return u, v


def negate_index(index: int):
    return index + 1 if index % 2 == 0 else index - 1


def classify(problem: SatProblem):
    g = nx.DiGraph()
    # print(problem)
    # print(problem.atoms)
    # atoms = sorted(problem.atoms)
    atoms = problem.atoms
    indices = Indices(atoms)
    g.add_nodes_from(range(2 * len(atoms) + 2))
    # print(len(g))
    # universals = sorted(problem.universals)
    for univ in problem.universals:
        u, v = indices[univ]
        # print(f"For {str(univ)} adding edge: {(u, v)} and {negate_index(v), negate_index(u)}")
        g.add_edge(u, v)
        # for all(-x,y) we cannot follow that all(x,-y)
        # if not (not univ.arg1.polarity and univ.arg2.polarity):
        g.add_edge(negate_index(v), negate_index(u))
    problem.graph = to_adj_list(g)
    shortest_paths = {(k, v): vv + 1 for k, d in all_pairs_bellman_ford_path_length(g) for v, vv in d.items()}
    # print(shortest_paths)
    # print(shortest_paths)
    # existentials = sorted(problem.existentials)
    inconsistencies = []
    for exis in problem.existentials:
        # print(exis)
        u, v = indices[exis]
        not_u = negate_index(u)
        not_v = negate_index(v)
        # print(u, v, not_u, not_v)
        u_not_v = shortest_paths.get((u, not_v), 0)
        u_not_u = shortest_paths.get((u, not_u), 0)
        v_not_v = shortest_paths.get((v, not_v), 0)
        if u_not_u + u_not_v + v_not_v > 0:
            # inconsistencies.append(min(x for x in [u_not_u, u_not_v, v_not_v] if x > 0))
            inconsistencies.extend(x for x in [u_not_u, u_not_v, v_not_v] if x > 0)
    problem.all_depths = sorted(set(inconsistencies))
    return min(inconsistencies) if inconsistencies else 0


def get_random_formulas_from_atoms(atoms: List[Atom], size: int) -> List[Formula]:
    args1 = random.choices(atoms, k=size)
    args2 = random.choices(atoms, k=size)
    pols = random.choices([True, False], k=size)
    quants = random.choices(['all', 'some'], k=size)
    return [Formula(quant, arg1, Literal(*lit)) for quant, arg1, *lit in zip(quants, args1, pols, args2)]


def get_random_problem(num_premises: int, num_vars: int, negate_subjects=False) -> SatProblem:
    atoms = [Atom(i) for i in range(num_vars)]
    args1 = random.choices(atoms, k=num_premises)
    args2 = random.choices(atoms, k=num_premises)
    pols = random.choices([True, False], k=num_premises)
    quants = random.choices(['all', 'some'], k=num_premises)
    if negate_subjects:
        pols_subj = random.choices([True, False], k=num_premises)
        formulas = [Formula(quant, Literal(pol1, arg1), Literal(pol2, arg2)) for quant, pol1, arg1, pol2, arg2 in
                    zip(quants, pols_subj, args1, pols, args2)]
    else:
        formulas = [Formula(quant, arg1, Literal(*lit)) for quant, arg1, *lit in zip(quants, args1, pols, args2)]
    return SatProblem.from_formulas(formulas, atoms)


def get_hard_problem(num_premises: int, num_vars: int, z: int, harden_ratio=0.2, seed=None) -> SatProblem:
    if seed:
        random.seed(seed)
    p = get_random_problem(num_premises, num_vars)
    ctr = count()

    def next_unique_atom() -> Generator[Atom, None, None]:
        used_atoms = set(a for f in (p.universals + p.existentials) for a in f.atoms)

        while True:
            atom = Atom(int(next(ctr)))
            if atom not in used_atoms:
                yield atom

    newniversals = []
    atoms = p.atoms
    for a in p.atoms:
        a.a = 'yes'
    hardens = random.choices([True, False], weights=[harden_ratio, 1 - harden_ratio], k=len(p.universals))
    for f, harden in zip(p.universals, hardens):
        assert f.arg1 in atoms and f.arg2.atom in atoms
        if harden:
            chain, _ = list(zip(*(zip(next_unique_atom(), range(z)))))
            # print([str(c) for c in chain])
            atoms.extend(chain)
            newniversals.append(Formula('all', f.arg1, Literal(True, chain[0])))
            for arg1, arg2 in zip(chain, map(lambda x: Literal(True, x), chain[1:])):
                newniversals.append(Formula('all', arg1, arg2))
            newniversals.append(Formula('all', chain[-1], f.arg2))
        else:
            newniversals.append(f)
    new_existentials = []
    for f in p.existentials:
        assert f.arg1 in atoms and f.arg2.atom in atoms
        if f.arg1 == f.arg2.atom and not f.arg2.polarity:
            chain, _ = zip(*(zip(next_unique_atom(), range(z))))
            # print([str(c) for c in chain])
            atoms.extend(chain)
            new_existentials.append(Formula('some', f.arg1, Literal(True, chain[0])))
            for arg1, arg2 in zip(chain, map(lambda x: Literal(True, x), chain[1:])):
                assert arg1 in atoms and arg2.atom in atoms
                newniversals.append(Formula('all', arg1, arg2))
            newniversals.append(Formula('all', chain[-1], f.arg2))

        else:
            new_existentials.append(f)
    new_p = SatProblem(newniversals, new_existentials, atoms)
    before_reorder = str(new_p)
    old_classify = classify(new_p)
    new_p.reorder(shuffle=True)
    new_p.depth = classify(new_p)
    assert old_classify == new_p.depth, f"\n{before_reorder}\n====\n{new_p}\n{old_classify} == {new_p.depth}"
    return new_p


def get_consistent_chain(chain: List[Atom], case=0):
    existential, univs = get_inconsistent_chain(chain, case)
    existential.arg2.polarity = not existential.arg2.polarity
    return existential, univs


def get_inconsistent_chain(chain: List[Atom], case=0) -> Tuple[Formula, List[Formula]]:
    univs = []

    if case == 0:
        existential = Formula('some', chain[0], Literal(False, chain[-1]))
        for arg1, arg2 in zip(chain, map(lambda x: Literal(True, x), chain[1:])):
            univs.append(Formula('all', arg1, arg2))
    elif case == 1:
        existential = Formula('some', chain[0], Literal(True, chain[-1]))
        for arg1, arg2 in zip(chain[:-2], map(lambda x: Literal(True, x), chain[1:])):
            univs.append(Formula('all', arg1, arg2))
        univs.append(Formula('all', chain[-2], Literal(False, chain[-1])))
    elif case == 2:
        assert len(chain) > 2, "For case 2 chain must be at least of 3 atoms!"
        # TODO: introduce a flipped chain all(x_i, -x_j) then all(x_n, x_n-1)...
        if random.choice([True, False]):
            existential = Formula('some', Literal(True, chain[0]), Literal(True, chain[-1]))
        else:
            existential = Formula('some', Literal(True, chain[-1]), Literal(True, chain[0]))
        flip_idx = random.choice(range(1, len(chain) - 1))
        # print(flip_idx)
        # first, do as in case 0
        _, normal_chain = get_consistent_chain(chain[:flip_idx], case=0)
        if random.choice([True, False]):
            flip = Formula('all', chain[flip_idx - 1], Literal(False, chain[flip_idx]))
        else:
            flip = Formula('all', chain[flip_idx], Literal(False, chain[flip_idx - 1]))
        # print(chain[:flip_idx:-1])
        _, reversed_chain = get_consistent_chain(chain[:flip_idx - 1:-1], case=0)
        univs.extend(normal_chain + [flip] + reversed_chain)
    elif case == 3:
        first_pol = random.choice([True, False])
        last_pol = first_pol
        for cur, next in zip(chain, chain[1:]):
            # if x0 == +
            if last_pol:
                subcase = random.choice(range(4))
                # either generate all(x0,x1) (stay pos)
                if subcase == 0:
                    univs.append(Formula('all', Literal(True, cur), Literal(True, next)))
                # or all(-x1, -x0) (stay pos)
                elif subcase == 1:
                    univs.append(Formula('all', Literal(False, next), Literal(False, cur)))
                # or all(x0, -x1) (swap to neg)
                elif subcase == 2:
                    univs.append(Formula('all', Literal(True, cur), Literal(False, next)))
                # or all(x1, -x0) (swap to neg)
                elif subcase == 3:
                    univs.append(Formula('all', Literal(True, next), Literal(False, cur)))
                last_pol = subcase in (0, 1)

            # if x0 == -
            else:
                subcase = random.choice(range(4))
                # either generate all(-x0, x1) (swap to pos)
                if subcase == 0:
                    univs.append(Formula('all', Literal(False, cur), Literal(True, next)))
                    # either generate all(-x1, x0) (swap to pos)
                if subcase == 1:
                    univs.append(Formula('all', Literal(False, next), Literal(True, cur)))
                # or all(x1,x0) (stay neg)
                elif subcase == 2:
                    univs.append(Formula('all', Literal(True, next), Literal(True, cur)))
                # or all(-x0, -x1) (stay neg)
                elif subcase == 3:
                    univs.append(Formula('all', Literal(False, cur), Literal(False, next)))
                last_pol = subcase in (0, 1)
        existential = Formula('some', Literal(first_pol, chain[0]), Literal(not last_pol, chain[-1]))
    else:
        raise NotImplementedError()
    return existential, univs


def get_balanced_with_guaranteed_depth(num_premises, num_vars, depth, is_consistent, seed=None, do_shuffle=True,
                                       realise=False, cases=(0, 1, 2, 3), nouns=None):
    if seed:
        random.seed(seed)
    if is_consistent:
        # p = get_random_problem(num_premises, num_vars)
        # while classify(p) > 0:
        #     p = get_random_problem(num_premises, num_vars)
        # assert classify(p) == 0
        # p.depth = 0
        p = get_guaranteed_depth(num_premises, num_vars, depth, consistent=True, do_shuffle=do_shuffle, realise=realise,
                                 cases=cases, nouns=nouns)
        assert classify(p) == 0
    else:
        p = get_guaranteed_depth(num_premises, num_vars, depth, do_shuffle=do_shuffle, realise=realise, cases=cases,
                                 nouns=nouns)
        assert classify(p) == depth
    return p


def get_guaranteed_depth(num_premises: int, num_vars: int, depth: int = 5, consistent=False, do_shuffle=True,
                         realise=False, cases=(0, 1, 2, 3), nouns=None):
    # 1 create inconsistent chain `c` of length `depth`
    # 2 get a consistent sat problem sharing atoms with inconsistent chain
    # 3 get all inconsistent paths
    # 4 remove universals not in `c` to break inconsistencies until depth of problem = d
    neg_subj = 3 in cases
    existentials = []
    universals = []
    atoms = [Atom(i) for i in range(num_vars)]
    chain = atoms[num_vars - depth:]
    if consistent:
        existential_of_chain, universals_of_chain = get_consistent_chain(chain, random.choice(cases))
        depth = 0
    else:
        existential_of_chain, universals_of_chain = get_inconsistent_chain(chain, random.choice(
            cases))  # random.choice([0, 1]))
    existentials.append(existential_of_chain)
    # print(existential, '\n'.join(str(u) for u in universals_of_chain), sep='\n')
    # print("--")
    universals.extend(universals_of_chain)
    p = SatProblem.from_formulas(existentials + universals, atoms=atoms)
    assert classify(p) == depth, f"{p}\n{classify(p)}"
    formulas = get_random_formulas_from_atoms(atoms, num_premises - len(universals) - 1)
    # get random satisfiable problem
    while classify(SatProblem.from_formulas(formulas)) > 0:
        # print('recreating...')
        formulas = get_random_formulas_from_atoms(atoms, num_premises - len(universals) - 1)

    existentials.extend(e for e in formulas if e.quantifier == 'some')
    universals.extend(u for u in formulas if u.quantifier == 'all')
    problem_so_far = SatProblem(universals, existentials, atoms)
    # print("problem so far")
    # print('\n'.join(str(u) for u in universals), sep='\n')
    # print('\n'.join(str(u) for u in existentials), sep='\n')
    # print("==")
    ensure_depth(problem_so_far, existential_of_chain, universals_of_chain, depth, neg_subj=neg_subj)
    assert classify(problem_so_far) == depth, classify(problem_so_far)

    problem_so_far.reorder(shuffle=do_shuffle)
    assert classify(problem_so_far) == depth
    problem_so_far.depth = depth
    if realise:
        problem_so_far.realise(nouns)
    return problem_so_far
    # assert depth in problem_so_far.all_depths, (depth, problem_so_far.all_depths)


def ensure_depth(problem_so_far: SatProblem, existential_of_chain: Formula, universals_of_chain: List[Formula],
                 depth: int, neg_subj):
    atoms = problem_so_far.atoms
    while classify(problem_so_far) != depth:
        # print(f"depth too low...: {classify(problem_so_far)}")
        # 3. get all inconsistencies
        inconsistencies = get_all_inconsistencies(atoms, problem_so_far, depth)

        # print("****")
        # print(f"inconsistencies shorter than {depth}")
        # print(inconsistencies)
        filtered_formulas = []
        formulas_flat = []
        for inconsistency, ex in inconsistencies:
            fmls = []
            alls = from_path(inconsistency, neg_subj)

            for a in alls:
                if a in problem_so_far.universals:
                    if a not in universals_of_chain:
                        fmls.append(a)
                        formulas_flat.append(a)
                else:
                    # print(f"Skipping {a} because reverse in problem!")
                    assert reverse(a) in problem_so_far.universals
            filtered_formulas.append((fmls, ex))

        # identify candidates to remove:
        # print(f"Filtered Formulas: {filtered_formulas}")

        ordered_choices = sorted(set(formulas_flat), key=lambda fla: sum(fla in flas for flas, _ in filtered_formulas),
                                 reverse=True)
        # print(f"Ordered Choices: {ordered_choices}")
        # assert False
        removeds = []
        removed_exs = []
        for ff, ex in filtered_formulas:
            assert ex != existential_of_chain or ff
            if ff:
                # if this is true, it's already removed
                to_remove = next((x for x in removeds if x in ff), False)
                if not to_remove:
                    to_remove = next((x for x in ordered_choices if x in ff), False)
                    if not to_remove:
                        # this shouldn't happen
                        assert False
                    removeds.append(to_remove)
                    problem_so_far.universals.remove(to_remove)
            else:
                if ex not in removed_exs:
                    problem_so_far.existentials.remove(ex)
                    removed_exs.append(ex)
        # print(problem_so_far)


def get_all_inconsistencies(atoms, problem_so_far, depth=0):
    g = from_adj_list(problem_so_far.graph)
    shortest_paths = {(k, v): vv + 1 for k, d in all_pairs_bellman_ford_path_length(g) for v, vv in d.items()}
    inconsistencies: List[Tuple[List[int], Formula]] = []
    indices = Indices(atoms)
    for exis in problem_so_far.existentials:
        # print(exis)
        u, v = indices[exis]
        not_u = negate_index(u)
        not_v = negate_index(v)
        # print(u, v, not_u, not_v)
        u_not_v = shortest_paths.get((u, not_v), 0)
        u_not_u = shortest_paths.get((u, not_u), 0)
        v_not_v = shortest_paths.get((v, not_v), 0)
        if u_not_v and (depth == 0 or u_not_v < depth):
            inconsistencies.append((nx.shortest_path(g, u, not_v), exis))
        if u_not_u and (depth == 0 or u_not_u < depth):
            inconsistencies.append((nx.shortest_path(g, u, not_u), exis))
        if v_not_v and (depth == 0 or v_not_v < depth):
            inconsistencies.append((nx.shortest_path(g, v, not_v), exis))
    return inconsistencies


def reverse(formula: Formula) -> Formula:
    assert formula.quantifier == 'all'
    if formula.arg1.polarity:
        if not formula.arg2.polarity:
            return Formula('all', arg1=Literal(True, Atom(formula.arg2.atom.order)),
                           arg2=Literal(False, Atom(formula.arg1.atom.order)))
        else:
            return Formula('all', Literal(False, Atom(formula.arg2.atom.order)),
                           arg2=Literal(False, Atom(formula.arg1.atom.order)))
    else:
        if not formula.arg2.polarity:
            return Formula('all', arg1=Literal(True, Atom(formula.arg2.atom.order)),
                           arg2=Literal(True, Atom(formula.arg1.atom.order)))
        else:
            return Formula('all', arg1=Literal(False, Atom(formula.arg2.atom.order)),
                           arg2=Literal(True, Atom(formula.arg1.atom.order)))


def from_path(path, neg_subj=False) -> List[Formula]:
    result = []
    for index_from, index_to in zip(path, path[1:]):
        if index_from % 2 == 0:
            if index_to % 2 == 0:
                # even-even
                # all(+from,+to)
                result.append(Formula('all', Atom(int(index_from / 2)), Literal(True, Atom(int(index_to / 2)))))
                if neg_subj:
                    # all(-to,-from)
                    result.append(Formula('all', Literal(False, Atom(int(index_to / 2))),
                                          Literal(False, Atom(int(index_from / 2)))))
            else:
                # even-odd
                # all(+from,-to)
                result.append(
                    Formula('all', Atom(int(index_from / 2)), Literal(False, Atom(int((index_to - 1) / 2)))))
                # all(+to,-from)
                result.append(
                    Formula('all', Atom(int((index_to - 1) / 2)), Literal(False, Atom(int(index_from / 2)))))
        else:
            if index_to % 2 == 0:
                # odd-even
                if neg_subj:
                    # all(-from,+to)
                    result.append(Formula('all', Literal(False, Atom(int((index_from - 1) / 2))),
                                          Literal(True, Atom(int(index_to / 2)))))
                    result.append(Formula('all', Literal(False, Atom(int(index_to / 2))),
                                          Literal(True, Atom(int((index_from - 1) / 2)))))
                else:
                    raise ValueError("Cant have odd->even if no negative subjects!")
            else:
                # odd-odd
                # all(+from,+to)
                result.append(
                    Formula('all', Atom(int((index_to - 1) / 2)), Literal(True, Atom(int((index_from - 1) / 2)))))
                if neg_subj:
                    # all(-to,-from)
                    result.append(
                        Formula('all', Literal(False, Atom(int((index_from) / 2))),
                                Literal(False, Atom(int((index_to - 1) / 2)))))
    return result
