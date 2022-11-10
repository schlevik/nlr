import random
from dataclasses import dataclass, field

from typing import List, Union


@dataclass(unsafe_hash=True)
class Atom:
    """
    Class for unary atoms.
    Contains the name of the atom and a method to convert to str.
    """
    name: str = field(init=True, compare=True, hash=True)
    noun: str = field(default=None, compare=False, hash=False)

    def __repr__(self):
        """
        Converts atom to string.
        Returns: String representation.
        """
        return self.name


@dataclass(unsafe_hash=True)
class Batom:
    """
    Class for binary atoms.
    Contains the name of the atom and a method to convert to str.
    """
    name: str = field(init=True, compare=True, hash=True)
    verb: str = field(default=None, compare=False, hash=False)

    def __repr__(self):
        """
        Converts binary atom to string.
        Returns: String representation.
        """
        return self.name


@dataclass(frozen=True)
class Literal:
    """
    Class for unary literals. Contains the atom and the polarity,
    a static method for checking if two literals are opposite,
    and methods for conversion to several representations.
    """
    polarity: bool
    atom: Atom

    def toTPTP(self, var: str):
        """
        Converts literal to TPTP format.

        Args:
            var: Variable to use in brackets.
                 Generally, Y if literal occurs in e-term, X otherwise.

        Returns: String with TPTP representation.
        """
        if self.polarity:
            return str(self.atom) + "(" + var + ")"
        return "~(" + str(self.atom) + "(" + var + "))"

    def toDFG(self, var: str):
        """
        Converts literal to DFG format.

        Args:
            var: Variable to use in brackets.
                 Generally, Y if literal occurs in e-term, X otherwise.

        Returns: String with DFG representation.
        """
        if self.polarity:
            return str(self.atom) + "(" + var + ")"
        return "not(" + str(self.atom) + "(" + var + "))"

    def neg(self):
        """
        Returns: Opposite literal.
        """
        return Literal(not self.polarity, self.atom)

    def __repr__(self):
        """
        Converts literal to string.
        Polarity expressed as +/-.
        Returns: String representation.
        """
        return ('+' if self.polarity else '-') + str(self.atom)


@dataclass(frozen=True)
class Bliteral:
    """
    Class for binary literals. Contains the binary atom and the polarity
    and methods for conversion to several representations.
    """
    polarity: bool
    batom: Batom

    def toTPTP(self):
        """
        Converts binary literal to TPTP format.
        Returns: String with TPTP representation.
        """
        if self.polarity:
            return str(self.batom) + "(X,Y)"
        return "~(" + str(self.batom) + "(X,Y))"

    def toDFG(self):
        """
        Converts binary literal to DFG format.
        Returns: String with DFG representation.
        """
        if self.polarity:
            return str(self.batom) + "(X,Y)"
        return "not(" + str(self.batom) + "(X,Y))"

    def neg(self):
        """
        Returns: Opposite literal.
        """
        return Bliteral(not self.polarity, self.batom)

    def __repr__(self):
        """
        Converts binary literal to string.
        Polarity expressed as +/-.
        Returns: String representation.
        """
        return ('+' if self.polarity else '-') + str(self.batom)


@dataclass(frozen=True)
class Eterm:
    """
    Class for e-terms. Contains the quantifier and the literals
    and methods for conversion to several representations.
    """
    quant: str
    arg1: Literal
    arg2: Bliteral

    def toTPTP(self):
        """
        Converts e-term to TPTP format.
        Returns: String with TPTP representation.
        """
        ans = ""
        if self.quant == "some":
            ans += "?[Y]:(" + self.arg1.toTPTP("Y")
            ans += "&" + self.arg2.toTPTP() + ")"
        else:
            ans += "![Y]:(" + self.arg1.toTPTP("Y")
            ans += "=>" + self.arg2.toTPTP() + ")"
        return ans

    def toDFG(self):
        """
        Converts e-term to DFG format.
        Returns: String with DFG representation.
        """
        ans = ""
        if self.quant == "some":
            ans += "exists([Y],and("
        else:
            ans += "forall([Y],implies("
        ans += self.arg1.toDFG("Y") + "," + self.arg2.toDFG() + "))"
        return ans

    def neg(self):
        """
        Returns: Opposite e-term.
        """
        return Eterm("every" if self.quant == "some" else "some",
                     self.arg1,
                     self.arg2.neg())

    def __repr__(self):
        """
        Converts e-term to string.
        Returns: String representation.
        """
        return self.quant + "(" + str(self.arg1) + "," + str(self.arg2) + ")"


@dataclass(frozen=True)
class Rfmla:
    """
    Class for relational formulae.
    Contains the quantifier, literal, and e-term
    and methods for conversion to several representations.
    """
    quant: str
    arg1: Literal
    arg2: Eterm

    def toTPTP(self):
        """
        Converts relational formula to TPTP format.
        Returns: String with TPTP representation.
        """
        ans = ""
        if self.quant == "some":
            ans += "?[X]:(" + self.arg1.toTPTP("X")
            ans += "&" + self.arg2.toTPTP() + ")"
        else:
            ans += "![X]:(" + self.arg1.toTPTP("X")
            ans += "=>" + self.arg2.toTPTP() + ")"
        return ans

    def toDFG(self):
        """
        Converts relational formula to DFG format.
        Returns: String with DFG representation.
        """
        ans = ""
        if self.quant == "some":
            ans += "formula(exists([X],and("
        else:
            ans += "formula(forall([X],implies("
        ans += self.arg1.toDFG("X") + "," + self.arg2.toDFG() + ")))."
        return ans

    def toEnglish(self, realise=False):
        """
        Converts relatoinal formula to an English sentence
        according to the glosses in 2009 paper.
        Returns: String with English sentence.
        """
        ans = ""
        atom1 = str(self.arg1.atom) if not realise else str(self.arg1.atom.noun)
        atom2 = str(self.arg2.arg1.atom) if not realise else str(self.arg2.arg1.atom.noun)
        verb = str(self.arg2.arg2.batom) if not realise else str(self.arg2.arg2.batom.verb)

        def verb_fix():
            vb = verb.split()[0]
            if vb.endswith('ss') or vb.endswith('ch'):
                return (verb + "es ") if len(verb.split()) == 1 else (
                            verb.split()[0] + 'es ' + ' '.join(verb.split()[1:]) + ' ')
            else:
                return (verb + "s ") if len(verb.split()) == 1 else (
                            verb.split()[0] + 's ' + ' '.join(verb.split()[1:]) + ' ')

        if self.quant == "some":
            ans += "Some "
            if not self.arg1.polarity:
                ans += "non-"
            ans += atom1 + " "
            if self.arg2.arg2.polarity:

                ans += verb_fix()
                ans += self.arg2.quant + " "
            else:
                if self.arg2.quant == "some":
                    ans += "does not " + verb + " every "
                else:
                    ans += verb_fix() + "no "

        else:
            ans += "Every " if self.arg2.arg2.polarity else "No "
            if not self.arg1.polarity:
                ans += "non-"
            # if verb.endswith("ch") or verb.endswith('ss'):
            #     verb = f"{verb}e"
            ans += atom1 + " " + verb_fix()
            if self.arg2.arg2.polarity:
                ans += self.arg2.quant + " "
            else:
                ans += "every " if self.arg2.quant == "some" else "any "
        if not self.arg2.arg1.polarity:
            ans += "non-"
        ans += atom2 + "."
        return ans

    def neg(self):
        """
        Returns: Opposite formula.
        """
        return Rfmla("every" if self.quant == "some" else "some",
                     self.arg1, self.arg2.neg())

    def __repr__(self):
        """
        Converts relational formula to string.
        Returns: String representation.
        """
        return self.quant + "(" + str(self.arg1) + "," + str(self.arg2) + ")"


@dataclass(frozen=True)
class Sfmla:
    """
    Class for non-relational formulae.
    Contains the quantifier and the literals
    and methods for conversion to several representations.
    """
    quant: str
    arg1: Literal
    arg2: Literal

    def toTPTP(self):
        """
        Converts formula to TPTP format.
        Returns: String with TPTP representation.
        """
        ans = ""
        if self.quant == "some":
            ans += "?[X]:(" + self.arg1.toTPTP("X")
            ans += "&" + self.arg2.toTPTP("X") + ")"
        else:
            ans += "![X]:(" + self.arg1.toTPTP("X")
            ans += "=>" + self.arg2.toTPTP("X") + ")"
        return ans

    def toDFG(self):
        """
        Converts formula to DFG format.
        Returns: String with DFG representation.
        """
        ans = ""
        if self.quant == "some":
            ans += "formula(exists([X],and("
        else:
            ans += "formula(forall([X],implies("
        ans += self.arg1.toDFG("X") + "," + self.arg2.toDFG("X") + ")))."
        return ans

    def toEnglish(self, realise=False):
        """
        Converts formula to an English sentence
        according to the glosses in 2009 paper.
        Returns: String with English sentence.
        """
        # ans = ""
        # if self.quant == "some":
        #     ans += "Some "
        #     if not self.arg1.polarity:
        #         ans += "non-"
        #     ans += str(self.arg2.atom) + " is "
        #     if not self.arg2.polarity:
        #         ans += "not "
        #     ans += "a " + str(self.arg2.atom) + "."
        # else:
        #     ans += "Every " if self.arg2.polarity else "No "
        #     if not self.arg1.polarity:
        #         ans += "non-"
        #     ans += str(self.arg1.atom) + " is a " + str(self.arg2.atom) + "."
        # return ans
        if self.quant == 'every':
            if self.arg2.polarity:
                quantifier = "Every"
            else:
                quantifier = "No"
        else:
            quantifier = "Some"
        det2 = "an" if (not realise or self.arg2.atom.noun[0] in 'aeio') else 'a'
        arg1 = str(self.arg1.atom) if not realise else self.arg1.atom.noun
        arg2 = str(self.arg2.atom) if not realise else self.arg2.atom.noun
        return f"{quantifier} {'non-' if not self.arg1.polarity else ''}{arg1} is{' not ' if self.quant == 'some' and not self.arg2.polarity else ' '}{det2} {arg2}."

    def neg(self):
        """
        Returns: Opposite formula.
        """
        return Sfmla("every" if self.quant == "some" else "some",
                     self.arg1, self.arg2.neg())

    def __repr__(self):
        """
        Converts formula to string.
        Returns: String representation.
        """
        return self.quant + "(" + str(self.arg1) + "," + str(self.arg2) + ")"


_verbs = None


def get_verbs() -> List[str]:
    global _verbs
    from pkg_resources import resource_string
    _verbs = _verbs or resource_string(__name__, "resources/verbs.txt").decode('utf-8').splitlines()
    return _verbs


@dataclass()
class Rproblem:
    """
    Class for problem in the extended relational syllogistic.
    Contains number of unary and binary atoms and formulae,
    as well as lists containing these, and a tag for consistency.
    If inconsistent, contains information on proof length and
    the number of formulae that need to be considered for refutation.
    Also contains a tag for easy/hard generation strategy
    and length of chains in case of hard.
    
    Contains a method for generating a random problem,
    a method for solving problems (assumes no negative subjects),
    and methods for conversion to several representations.
    """
    n_atoms: int
    n_batoms: int
    n_fmlas: int

    easy: bool  # true if random generation employed,
    # false if trivial absurdities removed
    # and universal formulae blown up into chains
    chain: int  # length of chains in case of hard generation

    atoms: List[Atom] = field(default_factory=list)
    batoms: List[Batom] = field(default_factory=list)
    fmlas: List[Union[Sfmla, Rfmla]] = field(default_factory=list)

    consistent = ""
    ref_subset = 0 # number of formulae one needs to look at for refutation
    proof_len = 0
    s_consistent = ""

    realised = False

    def realise(self, nouns=None, verbs=None):
        from nlr.generate import get_nouns
        nouns = nouns or get_nouns()
        verbs = verbs or get_verbs()
        for noun, atom in zip(random.sample(nouns, len(self.atoms)), self.atoms):
            atom.noun = noun
        for verb, batom in zip(random.sample(verbs, len(self.batoms)), self.batoms):
            batom.verb = verb
        self.realised = True

    def generate(self, prob_rel: float, prob_univ_sfmla: float,
                 prob_univ_rfmla: float, prob_univ_if_ex: float,
                 prob_univ_if_univ: float, prob_nsubj: float,
                 prob_nobj: float, prob_npred: float):
        """
        Generates a new problem based on the given probabilities
        and the required number of unary and binary atoms and formulae.

        Args:
            prob_rel: Probability of a formula being relational.
            prob_univ_sfmla: Probability of univ quantifier in s-formula.
            prob_univ_rfmla: Probability of univ quantifier in r-formula.
            prob_univ_if_ex: Probability of universal second quantifier
                             in existential r-formula.
            prob_univ_if_univ: Probability of universal second quantifier
                               in universal r-formula.
            prob_nsubj: Probability of negated subject in a formula.
            prob_nobj: Probability of negated object in a formula.
            prob_npred: Probability of negated predicate in a formula.
        """
        # Initialize lists of unary and binary atoms and number them.
        self.atoms = [Atom("p" + str(i)) for i in range(self.n_atoms)]
        self.batoms = [Batom("r" + str(i)) for i in range(self.n_batoms)]

        # List of new atoms added by universal chain blowup
        new_atoms = []

        # Generate random formulae according to assigned probability.
        types = random.choices([True, False],
                               [prob_rel, 1 - prob_rel],
                               k=self.n_fmlas)
        name = next_name()
        for i in range(self.n_fmlas):
            new_fmla = None
            if types[i]:
                new_fmla = random_rfmla(self.atoms, self.batoms,
                                        prob_univ_rfmla, prob_univ_if_ex,
                                        prob_univ_if_univ, prob_nsubj,
                                        prob_nobj, prob_npred)
            else:
                new_fmla = random_sfmla(self.atoms, prob_univ_sfmla,
                                        prob_nsubj, prob_npred)

                # If hard generation is employed, discard trivial absurdities
                if not self.easy:
                    while (new_fmla.arg1 == new_fmla.arg2.neg()):
                        new_fmla = random_sfmla(self.atoms, prob_univ_sfmla,
                                                prob_nsubj, prob_npred)

            # If hard generation is employed,
            # expand universal formulae into chains
            if not self.easy and self.chain and new_fmla.quant == "every":
                start = new_fmla.arg1
                end = new_fmla.arg2
                pol = random.choices([True, False],
                                     [1 - prob_nsubj, prob_nsubj])[0]
                prev_lit = Literal(pol, Atom(next(name)))
                new_atoms.append(prev_lit.atom)
                new_fmlas = [Sfmla("every", start, prev_lit)]
                for i in range(1, self.chain):
                    pol = random.choices([True, False],
                                         [1 - prob_nsubj, prob_nsubj])[0]
                    next_lit = Literal(pol, Atom(next(name)))
                    new_atoms.append(next_lit.atom)
                    new_fmlas.append(Sfmla("every", prev_lit, next_lit))
                    prev_lit = next_lit
                if isinstance(end, Eterm):
                    new_fmlas.append(Rfmla("every", prev_lit, end))
                else:
                    new_fmlas.append(Sfmla("every", prev_lit, end))
                self.fmlas.extend(new_fmlas)
            else:
                self.fmlas.append(new_fmla)

        # Amend problem attributes
        self.atoms.extend(new_atoms)
        self.n_atoms = len(self.atoms)
        self.n_fmlas = len(self.fmlas)

        # Shuffle list of formulae
        random.shuffle(self.fmlas)

    def solve(self):
        """
        Solves the problem.
        Assumes problem is in the non-extended relational syllogistic,
        i.e. there are no negative subjects and objects.

        Returns: 0 if problem is consistent;
                 1, 2, or 3 if problem is inconsistent
                 depending on type of inconsistency found.
        """
        # Iterate through all formulae in the problem
        # and create set of witnessess and adjacency list
        v0 = set()
        adj = {}
        for fmla in self.fmlas:
            if fmla.quant == "some":
                if ((fmla.arg1, fmla.arg2) not in v0
                        and (fmla.arg2, fmla.arg1) not in v0):
                    v0.add((fmla.arg1, fmla.arg2))
            else:
                if str(fmla.arg1) not in adj:
                    adj[str(fmla.arg1)] = []
                if str(fmla.arg2.neg()) not in adj:
                    adj[str(fmla.arg2.neg())] = []
                adj[str(fmla.arg1)].append(fmla.arg2)
                adj[str(fmla.arg2.neg())].append(fmla.arg1.neg())

        # Store the indeces of all items for quick access
        atom_idx = {str(atom): i for i, atom in enumerate(self.atoms)}

        # Matrix for cooccurrence of items in entailment
        # from witness (for last refutation condition)
        occ = [[False for j in range(self.n_atoms)]
               for i in range(self.n_atoms)]

        # Matrix for cooccurrence of contradictory c-terms
        # in entailment from witness, stored by atoms in first argument
        rch = [[False for j in range(self.n_atoms)]
               for i in range(self.n_atoms)]

        # Convert set of witnesses to list for iteration
        v0 = list(v0)
        v_singl = set()  # set of singletons added to witness list so far

        # Go through list of witnesses, looking for a refutation
        for v in v0:
            v_reached_atom = set()  # set of atoms reached
            v_reached_neglit = set()  # set of negative literals reached
            reached_cterms = []  # set of reached c-terms
            curr_reached = set()  # set of reached terms (all)
            for start in v:
                if start in curr_reached:
                    continue

                # If starting from a literal, check for refutation
                # and populate cooccurrence matrix
                if isinstance(start, Literal):
                    idx = atom_idx[str(start.atom)]
                    if start.polarity:
                        if idx in v_reached_neglit:
                            return 1
                        for a in v_reached_atom:
                            occ[idx][a] = True
                            occ[a][idx] = True
                        occ[idx][idx] = True
                        v_reached_atom.add(idx)
                    else:
                        if idx in v_reached_atom:
                            return 1
                        v_reached_neglit.add(idx)
                else:
                    reached_cterms.append(start)

                # Using dfs, determine all reachable terms
                # from current list of witnesses
                curr_reached.add(start)
                stack = [start]
                while stack:
                    curr = stack.pop(0)

                    if str(curr) not in adj:
                        continue

                    for o in adj[str(curr)]:
                        if o in curr_reached:
                            continue

                        # If current is literal, check for refutation
                        # and populate cooccurrence matrix
                        if isinstance(o, Literal):
                            idx = atom_idx[str(o.atom)]
                            if o.polarity:
                                if idx in v_reached_neglit:
                                    return 1
                                for a in v_reached_atom:
                                    occ[idx][a] = True
                                    occ[a][idx] = True
                                occ[idx][idx] = True
                                v_reached_atom.add(idx)
                            else:
                                if idx in v_reached_atom:
                                    return 1
                                v_reached_neglit.add(idx)
                        else:
                            reached_cterms.append(o)
                        curr_reached.add(o)
                        stack.append(o)

            # Go through reachable c-terms and look for refutation
            for term in reached_cterms:
                # If reachable c-term is existential c-term,
                # extend list of witnesses
                if term.quant == "some" and term.arg1 not in v_singl:
                    v_singl.add(term.arg1)
                    v0.append([term.arg1])

                for oterm in reached_cterms:
                    if oterm.quant == "some" or oterm.arg2 != term.arg2.neg():
                        continue

                    # If contradictory term is also reachable
                    q = term.arg1
                    o = oterm.arg1
                    if term.quant == "every":  # and both are universal,
                        idx = atom_idx[str(q.atom)]
                        oidx = atom_idx[str(o.atom)]

                        # check for refutation
                        #if occ[idx][oidx]:
                        #    return False

                        # and populate c-term cooccurrence matrix
                        rch[idx][oidx] = True
                        rch[oidx][idx] = True
                        continue

                    # If term is existential, check for path from atoms,
                    # resulting in refutation if found
                    if bPath(q, o, adj, atom_idx, {atom_idx[str(q.atom)]}):
                        return 2

        # For all pairs of atoms, check if they cooccurred and also if
        # contradictory universal c-terms involving them cooccurred,
        # in which case we have a refutation
        for i in range(self.n_atoms):
            for j in range(i, self.n_atoms):
                if occ[i][j] and rch[i][j]:
                    return 3

        # If no refutation has been found, the problem is consistent
        return 0

    def generate_all_all(self, path_len: int, neutralize: bool, prob_n_subj, prob_n_obj):
        """
        Generates an all-all inconsistency with given length of chains.

        Args:
            path_len: The length of the six chains in the inconsistency.
        """
        self.atoms = [Atom("p" + str(i)) for i in range(self.n_atoms)]
        self.batoms = [Batom("r" + str(i)) for i in range(self.n_batoms)]
        random.shuffle(self.atoms)

        p = path_len + 1
        o1 = p + path_len
        o2 = o1 + path_len
        q = o2 + path_len + 2
        o3 = q + path_len - 1

        self.fmlas.append(Sfmla("some",
                                Literal(True, self.atoms[0]),
                                Literal(True, self.atoms[1])))
        self.fmlas.extend(rChain(path_len, self.atoms, 1))
        self.fmlas.extend(sChain(path_len, p, o1, False, True, self.atoms, p))
        self.fmlas.extend(sChain(path_len, p, o2, False, True, self.atoms, o1))

        self.fmlas.append(Sfmla("some",
                                Literal(True, self.atoms[o2 + 1]),
                                Literal(True, self.atoms[o2 + 2])))
        self.fmlas.extend(rChain(path_len, self.atoms, o2 + 2))
        self.fmlas.extend(sChain(path_len, q, o1, True, True, self.atoms, q))
        self.fmlas.extend(sChain(path_len, q, o2, True, False, self.atoms, o3))

        n = len(self.fmlas)
        done = False
        while not done:
            self.fmlas = self.fmlas[:n]
            types = random.choices([True, False], [0.2, 0.8], k=n)
            for i in range(n):
                new_fmla = None
                if types[i]:
                    new_fmla = random_rfmla(self.atoms, self.batoms,
                                            0.8, 0.8, 0.8, prob_n_subj, prob_n_obj, 0.5)
                else:
                    new_fmla = random_sfmla(self.atoms, 0.8, 0.5, 0.5)

                    while new_fmla.arg1 == new_fmla.arg2.neg():
                        new_fmla = random_sfmla(self.atoms, 0.8, 0.5, 0.5)

                self.fmlas.append(new_fmla)

            done = self.solve() == 3
            if done and neutralize:
                self.fmlas[n - 1] = self.fmlas[n - 1].neg()
                done = not self.solve()
                if not done:
                    self.fmlas[n - 1] = self.fmlas[n - 1].neg()

        self.n_atoms = len(self.atoms)
        self.n_fmlas = len(self.fmlas)

        random.shuffle(self.fmlas)

    def toTPTP(self):
        """
        Converts problem to TPTP format.
        Returns: String with TPTP representation.
        """
        ans = ""
        for i, fmla in enumerate(self.fmlas):
            ans += "fof(formula" + str(i) + ",axiom," + fmla.toTPTP() + ").\n"
        return ans

    def toDFG(self):
        """
        Converts problem to DFG format.
        Returns: String with DFG representation.
        """
        ans = "begin_problem(relSylProblem).\n\nlist_of_descriptions.\n"
        ans += "name({*RandomProblemName*}).\n"
        ans += "author({*Ian Pratt-Hartmann*}).\nstatus(unsatisfiable).\n"
        ans += "description({*Problem taken from the easy list*}).\n"
        ans += "end_of_list.\n\nlist_of_symbols.\npredicates["
        for atom in self.atoms:
            ans += "(" + str(atom) + ",1),"
        for batom in self.batoms:
            ans += "(" + str(batom) + ",2)"
            if batom == self.batoms[self.n_batoms - 1]:
                ans += "].\n"
            else:
                ans += ","
        ans += "end_of_list.\n\nlist_of_formulae(axioms).\n"
        for fmla in self.fmlas:
            ans += fmla.toDFG() + "\n"
        ans += "end_of_list.\n\nlist_of_formulae(conjectures).\n"
        ans += "end_of_list.\n\nend_problem.\n"
        return ans

    def toEnglish(self, realise=False):
        """
        Converts problem to English.
        Returns: String with English representation.
        """
        if realise and not self.realised:
            self.realise()
        return "\n".join([fmla.toEnglish(realise) for fmla in self.fmlas]) + "\n"

    def toJSON(self, realise=False):
        """
        Converts problem to JSON format.
        Returns: Dictionary with JSON representation.
        """
        assert realise
        json = {"label": self.consistent,
                "sentence": self.toEnglish(realise),
                "easy": self.easy}
        if realise:
            json['raw'] = self.toEnglish()
        if not self.easy:
            json["chain"] = self.chain
        if self.consistent == "inconsistent":
            json["ref_subset"] = self.ref_subset
            json["proof_len"] = self.proof_len
            json["s_label"] = self.s_consistent
        return json

    def __repr__(self):
        """
        Converts problem to string.
        Returns: String representation.
        """
        return "\n".join([str(fmla) for fmla in self.fmlas]) + "\n"


def random_eterm(atoms: List[Atom], batoms: List[Batom],
                 prob_univ: float, prob_nobj: float, prob_npred: float):
    """
    Generates a random e-term.
    Quantifier is chosen according to the assigned probability.
    Literals are chosen randomly out of lists of atoms
    and assigned polarities according to the assigned probabilities.

    Args:
        atoms: List of atoms to choose for the unary literal.
        batoms: List of binary atoms to choose for the binary literals.
        prob_univ: Probability of quantifier being universal.
        prob_nobj: Probability of unary literal being negative.
        prob_npred: Probability of binary literal being negative.

    Returns: Random e-term.
    """
    quant = random.choices(["every", "some"], [prob_univ, 1 - prob_univ])[0]
    pol = random.choices([True, False], [1 - prob_nobj, prob_nobj])[0]
    arg1 = Literal(pol, random.choice(atoms))
    pol = random.choices([True, False], [1 - prob_npred, prob_npred])[0]
    arg2 = Bliteral(pol, random.choice(batoms))
    return Eterm(quant, arg1, arg2)


def random_rfmla(atoms: List[Atom], batoms: List[Batom], prob_univ: float,
                 prob_univ_if_ex: float, prob_univ_if_univ: float,
                 prob_nsubj: float, prob_nobj: float, prob_npred: float):
    """
    Generates a random relational formula.
    Quantifier is chosen according to the assigned probability.
    Literal is chosen randomly out of list of atoms
    and assigned a polarity according to the assigned probability.
    E-term is generated randomly.

    Args:
        atoms: List of atoms to choose for the unary literal and e-term.
        batoms: List of binary atoms to choose for the e-term.
        prob_univ: Probability of leading quantifier being universal.
        prob_univ_eterm: Probability of universal quantifier in e-term.
        prob_nsubj: Probability of literal being negative.
        prob_nobj: Probability of negative unary literal in e-term.
        prob_npred: Probability of negative binary literal in e-term.

    Returns: Random relational formula.
    """
    quant = random.choices(["every", "some"], [prob_univ, 1 - prob_univ])[0]
    pol = random.choices([True, False], [1 - prob_nsubj, prob_nsubj])[0]
    arg1 = Literal(pol, random.choice(atoms))
    if quant == "some":
        arg2 = random_eterm(atoms, batoms, prob_univ_if_ex,
                            prob_nobj, prob_npred)
    else:
        arg2 = random_eterm(atoms, batoms, prob_univ_if_univ,
                            prob_nobj, prob_npred)
    return Rfmla(quant, arg1, arg2)


def random_sfmla(atoms: List[Atom], prob_univ: float,
                 prob_nsubj: float, prob_npred: float):
    """
    Generates a random non-relational formula.
    Quantifier is chosen according to the assigned probability.
    Literals are chosen randomly out of list of atoms
    and assigned polarities according to the assigned probability.

    Args:
        atoms: List of atoms to choose for the literals.
        prob_univ: Probability of quantifier being universal.
        prob_nsubj: Probability of first literal being negative.
        prob_npred: Probability of second literal being negative.

    Returns: Random formula.
    """
    quant = random.choices(["every", "some"], [prob_univ, 1 - prob_univ])[0]
    pol = random.choices([True, False], [1 - prob_nsubj, prob_nsubj])[0]
    arg1 = Literal(pol, random.choice(atoms))
    pol = random.choices([True, False], [1 - prob_npred, prob_npred])[0]
    arg2 = Literal(pol, random.choice(atoms))
    return Sfmla(quant, arg1, arg2)


def next_name():
    """
    Generates next unique literal name.
    Returns: Next unique literal name.
    """
    num = 0
    while True:
        yield "pl" + str(num)
        num += 1


def bPath(start: Literal, end: Literal, adj: dict, idx: dict, reached: set):
    """
    Looks for a path from the starting atom
    to the destination atom using the adjacency list provided.

    Args:
        start: Starting atom.
        end: Destination atom.
        adj: Adjacency list using str representation of atoms as keys.
        idx: Dictionary linking atoms to their indeces.
        reached: Indeces of atoms already reached.

    Returns: True if there is such a path, False otherwise.
    """
    # If start atom and destination are the same, there is a path
    if start == end:
        return True

    # If start atom has no adjacent atoms, there is no path
    if str(start) not in adj:
        return False

    # If destination is adjacent to start atom, there is a path    
    if end in adj[str(start)]:
        return True

    # Look through all atom adjacent to start atom
    # for a path from them to destination
    for nxt in adj[str(start)]:
        if (isinstance(nxt, Literal)
                and nxt.polarity
                and idx[str(nxt.atom)] not in reached):
            new_reached = reached | {idx[str(nxt.atom)]}
            if bPath(nxt, end, adj, idx, new_reached):
                # If there is such a path, there is also a path as required.
                return True

    # If we have exhausted all adjacent atoms
    # and haven't found a path, no such path exists
    return False

def rChain(length: int, atoms: List[Atom], idx: int):
    """
    Produces an r-chain of formulas using atoms
    from the list starting from the given index.

    Args:
        length: Length of the r-chain.
        atoms: List of atoms to use in the chain.
        idx: Index to start from in the atom list.

    Returns: List of formulas forming the r-chain.
    """
    return [Sfmla("every",
                  Literal(True, atoms[idx + i]),
                  Literal(True, atoms[idx + i + 1]))
                if random.choice([True, False])
                else Rfmla("every",
                           Literal(True, atoms[idx + i]),
                           Eterm("some",
                                 Literal(True, atoms[idx + i + 1]),
                                 Bliteral(random.choice([True, False]),
                                          Batom("r0"))))
                    for i in range(length)]

def sChain(length: int, start: int, end: int, rel: bool,
           pol: bool, atoms: List[Atom], idx: int):
    """
    Produces an s-chain of formulas using atoms
    from the list starting from the given index.

    Args:
        length: Length of the s-chain.
        start: Index of starting atom in the chain.
        end: Index of destination atom in the chain.
        rel: Whether the final formula in the chain is relational.
        pol: Polarity of the binary literal in final formula
             if relational. No effect otherwise.
        atoms: List of atoms to use in the chain.
        idx: Index to start from in the atom list.

    Returns: List of formulas forming the s-chain.
    """
    stm = Literal(True, atoms[start])
    etm = Literal(True, atoms[end])
    if rel:
        etm = Eterm("every", etm, Bliteral(pol, Batom("r0")))
    if length == 1:
        return [Rfmla("every", stm, etm) if rel else Sfmla("every", stm, etm)]
    fmlas = [Sfmla("every", stm, Literal(True, atoms[idx + 1]))]
    fmlas.extend([Sfmla("every",
                        Literal(True, atoms[idx + i]),
                        Literal(True, atoms[idx + i + 1]))
                      for i in range(1, length - 1)])
    lit = Literal(True, atoms[idx + length - 1])
    fmlas.append(Rfmla("every", lit, etm) if rel else Sfmla("every", lit, etm))
    return fmlas
