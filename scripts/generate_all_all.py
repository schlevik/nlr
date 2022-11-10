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
    c_min: int = field(default=None)  # number of formulae
    af_ratio: float = field(default=0.75)
    bf_ratio: float = field(default=0.25)
    c_max: int = field(default=None)  # number of formulae
    prefix: str = field(default="")

    def __post_init__(self):
            assert self.c_min and self.c_max


def param_generator(script_args: ScriptArguments, tag, total_len, length):
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
    # if script_args.nf_min:
    #     print("With variable number of formulas!")
    #     ctr = count()
    #     for nf in range(script_args.nf_min, script_args.nf_max + 1):
    #         for _ in range(script_args.s):
    #             na = round(script_args.af_ratio * nf)
    #             nb = round(script_args.bf_ratio * nf)
    #             yield next(ctr), nf, na, nb, script_args, tag, nouns, verbs, next(seeds)
    # else:
    print("With fix number of formulas!")
    for i in range(script_args.s):
        yield i, length, script_args.nf, script_args.na, script_args.nb, script_args, tag, nouns, verbs, next(seeds)


def main():
    # Get command line arguments
    parser = HfArgumentParser((ScriptArguments,))
    args, *_ = parser.parse_args_into_dataclasses()
    args: ScriptArguments

    # Create name tag for problem set directory,
    # containing information about all parameters
    tag = "" if args.e else ("hard" + str(args.ch))
    tag += "na" + str(args.na) + "nb" + str(args.nb)
    tag += f"nfmin{args.c_min}nfmax{args.c_max}"
    tag += "pr" + str(args.pr) + "pus" + str(args.pus) + "puf" + str(args.puf)
    tag += "pue" + str(args.pue) + "puu" + str(args.puu)
    tag += "pns" + str(args.pns) + "pno" + str(args.pno) + "pnp" + str(args.pnp)
    tag += "s" + str(args.s)
    tag += "sd" + str(args.sd)
    name = tag
    tag = os.path.join(args.prefix, tag)
    os.makedirs(tag,exist_ok=True) # Create the directory
    total_len = args.s * (args.c_max+1 - args.c_min)

    # Seed the random number generator
    random.seed(args.sd)

    json_out = []
    for length in range(args.c_min, args.c_max+1):
        json_out.extend(Parallel(n_jobs=11)(delayed(generate)(*params) for params in tqdm(list(param_generator(args, tag, total_len, length)))))
    random.shuffle(json_out)

    proof_lens = [o['proof_len'] for o in json_out if o['label'] == 'inconsistent']  # List of proof lengths of unsatisfiable problems
    count = sum(o['label'] == 'consistent' for o in json_out)  # Variable to count number of satisfiable problems

    # Output all problems to JSON file
    with open(tag + "/" + name + ".json", "w") as f:
        f.write("\n".join(map(json.dumps, json_out)) + "\n")

    # Calculate percentage of satisfiable problems in set
    perc = count / total_len * 100

    print(f"{perc:.2f}% satisfiable.")

    # Create statistics file
    # with satisfiability percentage and proof length information
    with open(tag + "/stats.txt", "w") as f:
        f.write("Satisfiability percentage: " + str(perc) + "%\n\n")


def generate(i, n, nf, na, nb, args, tag, nouns, verbs, seed):
    random.seed(seed)
    # Store filepath for current problem
    out = tag + "/problem" + "_" + str(i + 1) + ".tptp"

    # Generate problem
    problem = Rproblem(6 * n + 3, 1, 12 * n + 4, args.e, args.ch)
    problem.generate_all_all(n, i % 2, args.pns, args.pno)

    result = problem.solve()
    assert result in [0, 3]
    result = "Refutation" if result else "Satisfiable"

    # Tag problem as satisfiable/unsatisfiable
    if result == "Refutation":  # If problem is unsatisfiable
        problem.consistent += "in"
    problem.consistent += "consistent"

    # Store problem's JSON representation
    if args.realise:
        problem.realise(nouns=nouns, verbs=verbs)
    return problem.toJSON(args.realise)


if __name__ == '__main__':
    main()
