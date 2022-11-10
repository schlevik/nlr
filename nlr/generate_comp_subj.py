import random
from dataclasses import dataclass, field

from typing import List

from nlr.generate_rel_problem import Atom
from nlr.generate_rel_problem import Literal


@dataclass()
class Xfmla:
    quant: str
    arg1: Atom
    arg2: Literal
    arg3: Literal

    def toTPTP(self):
        ans = ""
        if self.quant == "some":
            ans += "?[X]:((" + str(self.arg1) + "(X)&"
            ans += self.arg2.toTPTP("X") + ")&"
            ans += self.arg3.toTPTP("X") + ")"
        else:
            ans += "![X]:((" + str(self.arg1) + "(X)&"
            ans += self.arg2.toTPTP("X") + ")=>"
            ans += self.arg3.toTPTP("X") + ")"
        return ans

    def toEnglish(self, realise=False):
        arg1 = str(self.arg1) if not realise else self.arg1.noun
        arg2 = str(self.arg2.atom) if not realise else self.arg2.atom.noun
        arg3 = str(self.arg3.atom) if not realise else self.arg3.atom.noun
        ans = "Some " if self.quant == "some" else "Every "
        ans += arg1 + " who is "
        if not self.arg2.polarity:
            ans += "not "
        ans += "a " + arg2 + " is "
        if not self.arg3.polarity:
            ans += "not "
        ans += "a " + arg3 + "."
        return ans

    def __str__(self):
        return (self.quant + "(+" + str(self.arg1) + ","
                + str(self.arg2) + "," + str(self.arg3) + ")")


@dataclass()
class Xproblem:
    """
    Class for problem in the extended relational syllogistic.
    Contains number of unary and binary atoms and formulae,
    as well as lists containing these, and a tag for consistency.
    Also contains a method for generating a random problem,
    and methods for conversion to several representations.
    """
    n_atoms: int
    n_fmlas: int
    atoms: list
    fmlas: list
    consistent: str
    realised = False
    proof_len: int = field(init=False, default=None)

    def realise(self, nouns=None):
        from nlr.generate import get_nouns
        nouns = nouns or get_nouns()
        for noun, atom in zip(random.sample(nouns, len(self.atoms)), self.atoms):
            atom.noun = noun
        self.realised = True

    def generate(self, prob_univ: float,
                 prob_nclause: float, prob_npred: float):
        """
        Generates a new problem based on the given probabilities
        and the required number of unary and binary atoms and formulae.

        Args:
            prob_rel: Probability of a formula being relational.
            prob_univ_sfmla: Probability of univ quantifier in s-formula.
            prob_univ_rfmla: Probability of univ quantifier in r-formula.
            prob_univ_eterm: Probability of univ quantifier in e-term.
            prob_nsubj: Probability of negated subject in a formula.
            prob_nobj: Probability of negated object in a formula.
            prob_npred: Probability of negated predicate in a formula.
        """
        # Initialize lists of unary and binary atoms and number them.
        self.atoms = [Atom("p" + str(i)) for i in range(self.n_atoms)]

        self.fmlas = [random_xfmla(self.atoms, prob_univ,
                                   prob_nclause, prob_npred)
                      for i in range(self.n_fmlas)]

    def toTPTP(self):
        """
        Converts problem to TPTP format.
        Returns: String with TPTP representation.
        """
        ans = ""
        for i, fmla in enumerate(self.fmlas):
            ans += "fof(formula" + str(i) + ",axiom," + fmla.toTPTP() + ").\n"
        return ans

    def toEnglish(self, realise=False):
        """
        Converts problem to English.
        Returns: String with English representation.
        """
        if realise and not self.realised:
            self.realise()
        ans = ""
        for fmla in self.fmlas:
            ans += fmla.toEnglish(realise) + "\n"
        return ans

    def toJSON(self,realise=False):
        """
        Converts problem to JSON format.
        Returns: Dictionary with JSON representation.
        """
        output = {"label": self.consistent, "sentence": self.toEnglish(realise), "proof_len": self.proof_len}
        if realise:
            output['raw'] = self.toEnglish(realise=False)
        return output

    def __str__(self):
        """
        Converts problem to string.
        Returns: String representation.
        """
        ans = ""
        for fmla in self.fmlas:
            ans += str(fmla) + "\n"
        return ans


def random_xfmla(atoms: List[Atom], prob_univ: float,
                 prob_nclause: float, prob_npred: float):
    quant = random.choices(["every", "some"], [prob_univ, 1 - prob_univ])[0]
    pol = random.choices([True, False], [1 - prob_nclause, prob_nclause])[0]
    arg2 = Literal(pol, random.choice(atoms))
    pol = random.choices([True, False], [1 - prob_npred, prob_npred])[0]
    arg3 = Literal(pol, random.choice(atoms))
    return Xfmla(quant, random.choice(atoms), arg2, arg3)
