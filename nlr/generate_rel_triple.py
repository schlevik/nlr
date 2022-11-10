import random
from dataclasses import dataclass

from typing import List


@dataclass()
class Atom:
    """
    Class for unary atoms.
    Contains the name of the atom and a method to convert to str.
    """
    name: str

    def __str__(self):
        """
        Converts atom to string.
        Returns: String representation.
        """
        return self.name


@dataclass()
class Batom:
    """
    Class for binary atoms.
    Contains the name of the atom and a method to convert to str.
    """
    name: str

    def __str__(self):
        """
        Converts binary atom to string.
        Returns: String representation.
        """
        return self.name


@dataclass()
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
        return Literal(not self.polarity, self.atom)

    def __str__(self):
        """
        Converts literal to string.
        Polarity expressed as +/-.
        Returns: String representation.
        """
        return ('+' if self.polarity else '-') + str(self.atom)


@dataclass()
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
        return Bliteral(not self.polarity, self.batom)

    def __str__(self):
        """
        Converts binary literal to string.
        Polarity expressed as +/-.
        Returns: String representation.
        """
        return ('+' if self.polarity else '-') + str(self.batom)


@dataclass()
class Triple:
    arg1: Atom
    arg2: Atom
    arg3: Literal

    def toTPTP(self):
        return ("![X]:((" + str(self.arg1) + "(X)&" + str(self.arg2) + "(X))=>"
                + self.arg3.toTPTP("X") + ")")

    def toEnglish(self, o1: Literal, o2: Literal, r: Bliteral):
        t1 = str(self.arg1) + " " + str(r.batom) + "s "
        t1 = ("No " + t1 + "any") if r.polarity else ("Every " + t1 + "every")
        t1 += " " + ("non-" if o1.polarity else "") + str(o1.atom) + "."

        t2 = str(self.arg2) + " " + str(r.batom) + "s "
        t2 = ("No " + t2 + "any") if r.polarity else ("Every " + t2 + "every")
        t2 += (" " if o1.polarity else " non-") + str(o1.atom) + "."

        t3 = ("Every " if r.polarity else "No ")
        if self.arg3.polarity:
            t3 += "non-"
        t3 += str(self.arg3.atom) + " " + str(r.batom) + "s "
        t3 += "some " if r.polarity else "every "
        t3 += ("" if o2.polarity else "non-") + str(o2.atom) + "."

        return [t1, t2, t3]
                
    def __str__(self):
        return ("every(+" + str(self.arg1) + "&+" + str(self.arg2)
                + "," + str(self.arg3) + ")")


@dataclass()
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

    def toEnglish(self):
        """
        Converts formula to an English sentence
        according to the glosses in 2009 paper.
        Returns: String with English sentence.
        """
        ans = ""
        if self.quant == "some":
            ans += "Some "
            if not self.arg1.polarity:
                ans += "non-"
            ans += str(self.arg2.atom) + " is "
            if not self.arg2.polarity:
                ans += "not "
            ans += "a " + str(self.arg2.atom) + "."
        else:
            ans += "Every " if self.arg2.polarity else "No "
            if not self.arg1.polarity:
                ans += "non-"
            ans += str(self.arg1.atom) + " is a " + str(self.arg2.atom) + "."
        return ans

    def __str__(self):
        """
        Converts formula to string.
        Returns: String representation.
        """
        return self.quant + "(" + str(self.arg1) + "," + str(self.arg2) + ")"


@dataclass()
class Tproblem:
    """
    Class for problm in the extended relational syllogistic.
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
    easy: bool
    chain: int

    def generate(self, prob_triple: float, prob_univ: float, prob_neg: float):
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

        # List of new atoms added by universal chain blowup
        new_atoms = []

        # Generate random formulae according to assigned probability.
        types = random.choices(["sfmla", "triple"],
                               [1 - prob_triple, prob_triple],
                               k = self.n_fmlas)
        name = next_name()
        for i in range(self.n_fmlas):
            new_fmla = None
            if types[i] == "triple":
                new_fmla = random_triple(self.atoms, prob_neg)
            else:
                new_fmla = random_sfmla(self.atoms, prob_univ, 0, prob_neg)

                # If hard generation is employed, discard trivial absurdities
                if not self.easy:
                    while(new_fmla.arg1 == new_fmla.arg2.neg()):
                        new_fmla = random_sfmla(self.atoms, prob_univ,
                                                0, prob_neg)

            # If hard generation is employed,
            # expand universal formulae into chains
            if (not self.easy and self.chain
                    and (isinstance(new_fmla, Triple)
                        or new_fmla.quant == "every")):
                end = None
                prev_lit = Literal(True, Atom(next(name)))
                new_atoms.append(prev_lit.atom)
                if isinstance(new_fmla, Sfmla):
                    end = new_fmla.arg2
                    new_fmla = Sfmla("every", new_fmla.arg1, prev_lit)
                else:
                    end = new_fmla.arg3
                    new_fmla = Triple(new_fmla.arg1, new_fmla.arg2, prev_lit)
                new_fmlas = [new_fmla]
                for i in range(1, self.chain):
                    next_lit = Literal(True, Atom(next(name)))
                    new_atoms.append(next_lit.atom)
                    new_fmlas.append(Sfmla("every", prev_lit, next_lit))
                    prev_lit = next_lit
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

    def toTPTP(self):
        """
        Converts problem to TPTP format.
        Returns: String with TPTP representation.
        """
        ans = ""
        for i, fmla in enumerate(self.fmlas):
            ans += "fof(formula" + str(i) + ",axiom," + fmla.toTPTP() + ").\n"
        return ans

    def toEnglish(self):
        """
        Converts problem to English.
        Returns: String with English representation.
        """
        atom = next_atom()
        batom = next_batom()
        sentences = []
        for fmla in self.fmlas:
            if isinstance(fmla, Sfmla):
                sentences.append(fmla.toEnglish())
            else:
                o1 = Literal(random.choice([True, False]), next(atom))
                o2 = Literal(random.choice([True, False]), next(atom))
                r = Bliteral(random.choice([True, False]), next(batom))
                sentences.extend(fmla.toEnglish(o1, o2, r))
        random.shuffle(sentences)
        return "\n".join(sentences) + "\n"

    def toJSON(self):
        """
        Converts problem to JSON format.
        Returns: Dictionary with JSON representation.
        """
        return {"label": self.consistent, "sentence": self.toEnglish()}

    def __str__(self):
        """
        Converts problem to string.
        Returns: String representation.
        """
        return "\n".join(self.fmlas) + "\n"


def random_triple(atoms: List[Atom], prob_neg: float):
    pol = random.choices([True, False], [1 - prob_neg, prob_neg])[0]
    arg3 = Literal(pol, random.choice(atoms))
    return Triple(random.choice(atoms), random.choice(atoms), arg3)


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

def next_atom():
    """
    Generates next unique literal name.
    Returns: Next unique literal name.
    """
    num = 0
    while True:
        yield "o" + str(num)
        num += 1

def next_batom():
    num = 0
    while True:
        yield "r" + str(num)
        num += 1