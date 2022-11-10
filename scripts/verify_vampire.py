import os
import time
from dataclasses import field
from itertools import count

from joblib import Parallel, delayed
from transformers import HfArgumentParser

import re
import subprocess
import json
import numpy

from nlr.generate_rel_problem import *
from tqdm import trange, tqdm


@dataclass()
class ScriptArguments:
    """
    Class for command-line arguments.
    Contains all variable parameters used in generation of problem sets.
    """
    na: int = field(default=200)  # number of atoms
    nb: int = field(default=2)  # number of binary atoms
    nf: int = field(default=200)  # number of formulae
    pr: float = field(default=0.2)  # probability of relational
    pus: float = field(default=0.8)  # probability of universal in s-formula
    puf: float = field(default=0.8)  # probability of leading universal
    # in r-formula
    pue: float = field(default=0.8)  # probability of universal second
    # quantifier in existential r-formula
    puu: float = field(default=0.8)  # probability of universal second
    # quantifier in universal r-formula
    pns: float = field(default=0.5)  # probability of negated subject
    pno: float = field(default=0.5)  # probability of negated object
    pnp: float = field(default=0.5)  # probability of negated predicate
    e: bool = field(default=True)  # true for random generation,
    # false for removal of trivial absurdities
    # and universal chain blowup
    ch: int = field(default=0)  # length of chain blowup in hard generation
    s: int = field(default=500)  # how many problems to generate
    sd: int = field(default=random.randint(0, 9223372036854775807))  # seed
    realise: bool = field(default=False)
    nouns: str = field(default='nlr/resources/nouns.txt')
    verbs: str = field(default='nlr/resources/verbs.txt')
    nf_min: int = field(default=None)  # number of formulae
    af_ratio: float = field(default=0.75)
    bf_ratio: float = field(default=0.25)
    nf_max: int = field(default=None)  # number of formulae
    prefix: str = field(default="")

    def __post_init__(self):
        if self.nf_min or self.nf_max:
            assert self.nf_min and self.nf_max


def param_generator(script_args: ScriptArguments, tag, total_len):
    if script_args.realise:
        assert script_args.verbs and script_args.nouns
        with open(script_args.verbs) as f:
            verbs = f.read().splitlines()
        with open(script_args.nouns) as f:
            nouns = f.read().splitlines()
    else:
        nouns, verbs = [], []

    print(f"Generating {total_len} problems.")
    seeds = iter(random.sample(range(1000 * total_len), total_len))
    if script_args.nf_min:
        print("With variable number of formulas!")
        ctr = count()
        for nf in range(script_args.nf_min, script_args.nf_max + 1):
            for _ in range(script_args.s):
                na = round(script_args.af_ratio * nf)
                nb = round(script_args.bf_ratio * nf)
                yield next(ctr), nf, na, nb, script_args, tag, nouns, verbs, next(seeds)
    else:
        print("With fix number of formulas!")
        for i in range(script_args.s):
            yield i, script_args.nf, script_args.na, script_args.nb, script_args, tag, nouns, verbs, next(seeds)


def main():
    # Get command line arguments
    parser = HfArgumentParser((ScriptArguments,))
    args, *_ = parser.parse_args_into_dataclasses()
    args: ScriptArguments

    # Create name tag for problem set directory,
    # containing information about all parameters
    tag = "" if args.e else ("hard" + str(args.ch))
    tag += "na" + str(args.na) + "nb" + str(args.nb)
    tag += "nf" + str(args.nf) if not args.nf_min else f"nfmin{args.nf_min}nfmax{args.nf_max}"
    tag += "pr" + str(args.pr) + "pus" + str(args.pus) + "puf" + str(args.puf)
    tag += "pue" + str(args.pue) + "puu" + str(args.puu)
    tag += "pns" + str(args.pns) + "pno" + str(args.pno) + "pnp" + str(args.pnp)
    tag += "s" + str(args.s)
    tag += "sd" + str(args.sd)
    name = tag
    tag = os.path.join(args.prefix, tag)
    os.makedirs(tag,exist_ok=True) # Create the directory
    total_len = (args.nf_max - args.nf_min + 1) * args.s if args.nf_min else args.s


    # Seed the random number generator
    random.seed(args.sd)

    # List of problems in JSON format
    json_out = Parallel(n_jobs=11)(delayed(generate)(*params) for params in tqdm(list(param_generator(args, tag, total_len))))
    json_out = [o for o in json_out if o]

    proof_lens = [o['proof_len'] for o in json_out if o['label'] == 'inconsistent']  # List of proof lengths of unsatisfiable problems
    count = sum(o['label'] == 'consistent' for o in json_out)  # Variable to count number of satisfiable problems
    s_count = sum(("s_label" in o and o['s_label'] == "inconsistent") for o in json_out)

    # Output all problems to JSON file
    with open(tag + "/" + name + ".json", "w") as f:
        f.write("\n".join(map(json.dumps, json_out)) + "\n")

    # Calculate percentage of satisfiable problems in set
    perc = count / total_len * 100
    if perc != 100:
        s_perc = s_count / (total_len - count) * 100

    print(f"{perc:.2f}% satisfiable.")
    if perc != 100:
        print(f"{s_perc:.2f}% of unsatisfiable remain so if only s-fmlas considered.")

    # Create statistics file
    # with satisfiability percentage and proof length information
    with open(tag + "/stats.txt", "w") as f:
        f.write("Satisfiability percentage: " + str(perc) + "%\n")
        if perc != 100:
            f.write("Percentage of unsat fmlas remaining unsat if only s-fmlas considered: " + str(s_perc) + "%\n\n"
                    + "Proof lengths of unsat problems: \n" + str(proof_lens) + "\n"
                    + "Average proof length: "
                    + str(sum(proof_lens) / len(proof_lens)) + "\n"
                    + "Standard deviation: " + str(numpy.std(proof_lens)) + "\n")


regex = re.compile("(?<=Termination reason: )\w+(?=\\n)")
def generate(i, nf, na, nb, args, tag, nouns, verbs, seed):
    random.seed(seed)
    # Store filepath for current problem
    out = tag + "/problem" + "_" + str(i + 1) + ".tptp"

    # Generate problem
    problem = Rproblem(na, nb, nf, args.e, args.ch)
    problem.generate(args.pr, args.pus, args.puf, args.pue,
                     args.puu, args.pns, args.pno, args.pnp)

    # Write problem to file
    with open(out, "w") as f:
        f.write(problem.toTPTP())
    # Run Vampire on problem to check satisfiability
    # proof = subprocess.run(["./vampire", out],
    #                        capture_output=True,
    #                        text=True)
    with subprocess.Popen(["./vampire", out], stdout=subprocess.PIPE) as proc:
        output = proc.stdout.read().decode('utf-8')

    # Extract result
    try:

        result = re.search(regex, output).group()
    except AttributeError:
        print("I dieded. I is ded.")
        return None

    # Tag problem as satisfiable/unsatisfiable
    if result == "Refutation":  # If problem is unsatisfiable
        problem.consistent += "in"

        # record number of formulas required for refutation
        #problem.ref_subset = proof.stdout.count("[input]")
        # problem.ref_subset = output.count("[input]")
        # record the length of the proof
        # (subtract 14 lines of constant non-proof Vampire output)
        problem.proof_len = output.count("\n") - 14
        problem.ref_subset = output.count("[input]")

        s_out = out + " (s)"
        s_fmlas = [fmla for fmla in problem.fmlas
                       if isinstance(fmla, Sfmla)]
        s_problem = Rproblem(na, nb, nf, args.e, args.ch,
                             problem.atoms, problem.batoms, s_fmlas)

        with open(s_out, "w") as f:
            f.write(s_problem.toTPTP())

        with subprocess.Popen(["./vampire", s_out], stdout=subprocess.PIPE) as proc:
            s_output = proc.stdout.read().decode('utf-8')

        try:
            s_result = re.search(regex, s_output).group()
        except:
            raise RuntimeError()

        if s_result == "Refutation":
            problem.s_consistent += "in"
        problem.s_consistent += "consistent"
    else:  # If problem is satisfiable
        # count += 1  # uptick counter
        ...
    problem.consistent += "consistent"

    # Append tag(s) to file
    with open(out, "a") as f:
        f.write("% " + result + "\n")
        if result == "Refutation":
            f.write("% Number of lines in proof: "
                    + str(problem.proof_len) + "\n"
                    + "% S-fmlas only result: " + s_result + "\n")

    # Store problem's JSON representation
    if args.realise:
        problem.realise(nouns=nouns, verbs=verbs)
    return problem.toJSON(args.realise)


if __name__ == '__main__':
    main()
