import os
import random
import time
from dataclasses import field
from dataclasses import dataclass
from itertools import count

from joblib import Parallel, delayed
from transformers import HfArgumentParser

import re
import subprocess
import json
import numpy

from nlr.generate_comp_subj import Xproblem
from tqdm import trange, tqdm


@dataclass()
class ScriptArguments:
    """
    Class for command-line arguments.
    Contains all variable parameters used in generation of problem sets.
    """
    """
    Class for command-line arguments.
    Contains all variable parameters used in generation of problem sets.
    """
    na: int = field(default=200)  # number of atoms
    nf: int = field(default=200)  # number of formulae
    pu: float = field(default=0.8)  # probability of universal
    pnc: float = field(default=0.5)  # probability of negated clause
    pnp: float = field(default=0.5)  # probability of negated predicate
    s: int = field(default=500)  # how many problems to generate
    sd: int = field(default=random.randint(0, 9223372036854775807))  # seed
    realise: bool = field(default=False)
    nouns: str = field(default='nlr/resources/nouns.txt')
    nf_min: int = field(default=None)  # number of formulae min
    af_ratio: float = field(default=0.75)
    nf_max: int = field(default=None)  # number of formulae max
    prefix: str = field(default="")

    def __post_init__(self):
        if self.nf_min or self.nf_max:
            assert self.nf_min and self.nf_max


def param_generator(script_args: ScriptArguments, tag, total_len):
    if script_args.realise:
        assert script_args.nouns
        with open(script_args.nouns) as f:
            nouns = f.read().splitlines()
    else:
        nouns = []

    print(f"Generating {total_len} problems.")
    seeds = iter(random.sample(range(1000 * total_len), total_len))
    if script_args.nf_min:
        print("With variable number of formulas!")
        ctr = count()
        for nf in range(script_args.nf_min, script_args.nf_max + 1):
            for _ in range(script_args.s):
                na = round(script_args.af_ratio * nf - (nf - script_args.nf_min) * 0.015)
                # (i, nf, na, args, tag, nouns, seed)
                yield next(ctr), nf, na, script_args, tag, nouns, next(seeds)
    else:
        print("With fix number of formulas!")
        for i in range(script_args.s):
            # (i, nf, na, args, tag, nouns, seed)
            yield i, script_args.nf, script_args.na, script_args, tag, nouns, next(seeds)


def main():
    # Get command line arguments
    parser = HfArgumentParser((ScriptArguments,))
    args, *_ = parser.parse_args_into_dataclasses()
    args: ScriptArguments

    # Create name tag for problem set directory,
    # containing information about all parameters
    tag = "CompSubjna" + str(args.na)
    tag += "nf" + str(args.nf) if not args.nf_min else f"nfmin{args.nf_min}nfmax{args.nf_max}"
    tag += "pu" + str(args.pu) + "pnc" + str(args.pnc) + "pnp" + str(args.pnp)
    tag += "s" + str(args.s) + "sd" + str(args.sd)
    name = tag
    tag = os.path.join(args.prefix, tag)
    os.makedirs(tag, exist_ok=True)  # Create the directory
    total_len = (args.nf_max - args.nf_min + 1) * args.s if args.nf_min else args.s

    # Seed the random number generator
    random.seed(args.sd)

    # List of problems in JSON format
    json_out = Parallel(n_jobs=11)(
        delayed(generate)(*params) for params in tqdm(list(param_generator(args, tag, total_len))))
    json_out = [o for o in json_out if o]

    proof_lens = [o['proof_len'] for o in json_out if
                  o['label'] == 'inconsistent']  # List of proof lengths of unsatisfiable problems
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
        if perc != 100:
            f.write("Proof lengths of unsat problems: \n" + str(proof_lens) + "\n"
                    + "Average proof length: "
                    + str(sum(proof_lens) / len(proof_lens)) + "\n"
                    + "Standard deviation: " + str(numpy.std(proof_lens)) + "\n")


regex = re.compile("(?<=Termination reason: )\w+(?=\\n)")


def generate(i, nf, na, args, tag, nouns, seed):
    random.seed(seed)
    # Store filepath for current problem
    out = tag + "/problem" + "_" + str(i + 1) + ".tptp"

    # Generate problem
    problem = Xproblem(na, nf, [], [], "")
    problem.generate(args.pu, args.pnc, args.pnp)

    # Write problem to file
    with open(out, "w") as f:
        f.write(problem.toTPTP())

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
        # problem.ref_subset = proof.stdout.count("[input]")
        # problem.ref_subset = output.count("[input]")
        # record the length of the proof
        # (subtract 14 lines of constant non-proof Vampire output)
        problem.proof_len = output.count("\n") - 14
        # problem.ref_subset = output.count("[input]")
        # proof_lens.append(problem.proof_len)
    else:  # If problem is satisfiable
        # count += 1  # uptick counter
        ...
    problem.consistent += "consistent"

    # Append tag(s) to file
    with open(out, "a") as f:
        f.write("% " + result + "\n")
        if result == "Refutation":
            f.write("% Number of lines in proof: "
                    + str(problem.proof_len) + "\n")

    # Store problem's JSON representation
    if args.realise:
        problem.realise(nouns=nouns)
    return problem.toJSON(args.realise)


if __name__ == '__main__':
    main()
